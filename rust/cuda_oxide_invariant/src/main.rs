// SPDX-License-Identifier: Apache-2.0
//! Shape-invariant reference kernels for speculative target verification.
//!
//! These kernels intentionally assign one thread to one output element and
//! retain a fixed scalar reduction order.  Running M verification positions in
//! one launch therefore performs exactly the same arithmetic as M one-position
//! launches.  They are the correctness oracle for faster fixed-tree kernels.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use half::bf16;

#[cuda_module]
mod kernels {
    use super::*;

    /// C[M,N] = BF16(FP32(A[M,K]) @ FP32(B[K,N]) + bias[N]).
    ///
    /// K is always accumulated from zero to K-1 by the owning thread.  Neither
    /// M nor the grid shape participates in the arithmetic reduction tree.
    #[kernel]
    pub fn bf16_linear_fixed_k(
        m: u32,
        n: u32,
        k: u32,
        a: &[bf16],
        b: &[bf16],
        bias: &[bf16],
        has_bias: u32,
        mut c: DisjointSlice<bf16, thread::Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        if row >= m as usize || col >= n as usize {
            return;
        }
        let Some(out_idx) = (unsafe { thread::index_2d_runtime(n as usize) }) else {
            return;
        };

        let mut sum = if has_bias != 0 {
            bias[col].to_f32()
        } else {
            0.0
        };
        let mut inner = 0usize;
        while inner < k as usize {
            // Keep this scalar statement stable.  A tiled implementation must
            // reproduce this exact FP32 accumulation tree before promotion.
            sum += a[row * k as usize + inner].to_f32() * b[inner * n as usize + col].to_f32();
            inner += 1;
        }
        if let Some(dst) = c.get_mut(out_idx) {
            *dst = bf16::from_f32(sum);
        }
    }
}

fn input_value(i: usize) -> bf16 {
    bf16::from_f32((((i * 37 + 11) % 257) as f32 - 128.0) / 64.0)
}

fn run(ctx: &CudaContext, m: usize, n: usize, k: usize, a: &[bf16], b: &[bf16]) -> Vec<bf16> {
    let stream = ctx.default_stream();
    let a_dev = DeviceBuffer::from_host(&stream, a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, b).unwrap();
    let bias_dev = DeviceBuffer::from_host(&stream, &vec![bf16::ZERO; n]).unwrap();
    let mut c_dev = DeviceBuffer::from_host(&stream, &vec![bf16::ZERO; m * n]).unwrap();
    let module = kernels::load(ctx).expect("load embedded CUDA Oxide PTX");
    let block = 16u32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block), (m as u32).div_ceil(block), 1),
        block_dim: (block, block, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.bf16_linear_fixed_k(
            stream.as_ref(),
            cfg,
            m as u32,
            n as u32,
            k as u32,
            &a_dev,
            &b_dev,
            &bias_dev,
            0,
            &mut c_dev,
        )
    }
    .unwrap();
    c_dev.to_host_vec(&stream).unwrap()
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    // AudEx hidden size plus deliberately awkward N catches tail indexing.
    let (max_m, n, k) = (7usize, 2701usize, 2688usize);
    let a: Vec<_> = (0..max_m * k).map(input_value).collect();
    let b: Vec<_> = (0..k * n).map(|i| input_value(i + 19)).collect();

    for m in [1usize, 3, 5, 7] {
        let batched = run(&ctx, m, n, k, &a[..m * k], &b);
        let mut sequential = Vec::with_capacity(m * n);
        for row in 0..m {
            sequential.extend(run(&ctx, 1, n, k, &a[row * k..(row + 1) * k], &b));
        }
        assert_eq!(batched.len(), sequential.len());
        for (i, (lhs, rhs)) in batched.iter().zip(&sequential).enumerate() {
            assert_eq!(lhs.to_bits(), rhs.to_bits(), "M={m}, output={i}");
        }
        println!(
            "PASS M={m}: {} BF16 outputs are bitwise sequential-equivalent",
            m * n
        );
    }
}
