# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

class VerificationGroup(NamedTuple):
    tokens: tuple[int, ...]
    requests: tuple[int, ...]
    remaining: tuple[int, ...]


_groups: ContextVar[tuple[VerificationGroup, ...]] = ContextVar(
    "audex_invariant_groups", default=()
)


def groups() -> tuple[VerificationGroup, ...]:
    return _groups.get()


@contextmanager
def suspend_groups():
    token = _groups.set(())
    try:
        yield
    finally:
        _groups.reset(token)


@contextmanager
def verification_groups(batch: "InputBatch"):
    enabled = any(
        os.getenv(name) == "1"
        for name in (
            "VLLM_AUDEX_INVARIANT_ATTENTION",
            "VLLM_AUDEX_INVARIANT_MOE",
            "VLLM_AUDEX_INVARIANT_ROUTER",
            "VLLM_AUDEX_INVARIANT_TOPK",
        )
    )
    if not enabled or batch.num_draft_tokens == 0:
        yield
        return

    roles: dict[str, dict[str, int]] = {}
    for index, request_id in enumerate(batch.req_ids):
        if request_id.endswith("-cond") or "-cond-" in request_id:
            roles.setdefault(request_id.rsplit("-cond", 1)[0], {})["cond"] = index
        elif request_id.endswith("-uncond") or "-uncond-" in request_id:
            roles.setdefault(request_id.rsplit("-uncond", 1)[0], {})["uncond"] = index

    grouped: list[VerificationGroup] = []
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
            group = VerificationGroup(
                (int(starts[cond] + position), int(starts[uncond] + position)),
                (cond, uncond),
                (cond_len - position - 1, uncond_len - position - 1),
            )
            grouped.append(group)
            covered.update(group.tokens)

    for request in range(batch.num_reqs):
        length = int(starts[request + 1] - starts[request])
        for position in range(length):
            index = int(starts[request] + position)
            if index not in covered:
                grouped.append(
                    VerificationGroup(
                        (index,), (request,), (length - position - 1,)
                    )
                )
    grouped.sort(key=lambda group: min(group.tokens))
    token = _groups.set(tuple(grouped))
    try:
        yield
    finally:
        _groups.reset(token)
