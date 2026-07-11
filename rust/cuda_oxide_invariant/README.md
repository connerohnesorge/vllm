# CUDA Oxide invariant verification kernels

This crate pins NVlabs/cuda-oxide at `29396b7f643b1d42eb4d80b7347ad27bb011525a`.
It supplies the fixed-reduction correctness oracle used while making AudEx
multi-position speculative verification sequential-equivalent.

Build and prove proposal widths 1, 3, 5, and 7 on a CUDA 12+ Linux builder:

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide \
  --rev 29396b7f643b1d42eb4d80b7347ad27bb011525a cargo-oxide
cd rust/cuda_oxide_invariant
cargo oxide build --arch sm_89 -- --manifest-path Cargo.toml
./target/debug/vllm-cuda-oxide-invariant
```

The proof compares raw BF16 bits, not tolerances. Faster kernels may replace
the reference only after producing the same bits for every tested width.
