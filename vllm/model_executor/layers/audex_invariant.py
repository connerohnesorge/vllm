# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

_groups: ContextVar[tuple[tuple[int, ...], ...]] = ContextVar(
    "audex_invariant_groups", default=()
)


def groups() -> tuple[tuple[int, ...], ...]:
    return _groups.get()


@contextmanager
def verification_groups(batch: "InputBatch"):
    enabled = os.getenv("VLLM_AUDEX_INVARIANT_MOE") == "1"
    if not enabled or not batch.num_draft_tokens:
        yield
        return

    roles: dict[str, dict[str, int]] = {}
    for index, request_id in enumerate(batch.req_ids):
        if request_id.endswith("-cond"):
            roles.setdefault(request_id[:-5], {})["cond"] = index
        elif request_id.endswith("-uncond"):
            roles.setdefault(request_id[:-7], {})["uncond"] = index

    grouped: list[tuple[int, ...]] = []
    covered: set[int] = set()
    starts = batch.query_start_loc_np
    for pair in roles.values():
        if pair.keys() < {"cond", "uncond"}:
            continue
        cond, uncond = pair["cond"], pair["uncond"]
        cond_len = int(starts[cond + 1] - starts[cond])
        uncond_len = int(starts[uncond + 1] - starts[uncond])
        if cond_len != uncond_len:
            raise RuntimeError("CFG verification rows have unequal lengths")
        for position in range(cond_len):
            group = (int(starts[cond] + position), int(starts[uncond] + position))
            grouped.append(group)
            covered.update(group)

    grouped.extend((index,) for index in range(batch.num_tokens) if index not in covered)
    grouped.sort(key=min)
    token = _groups.set(tuple(grouped))
    try:
        yield
    finally:
        _groups.reset(token)
