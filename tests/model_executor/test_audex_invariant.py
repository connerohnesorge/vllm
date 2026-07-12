from types import SimpleNamespace

import numpy as np
import pytest

from vllm.model_executor.layers.audex_invariant import groups, verification_groups


def batch(starts, drafts):
    return SimpleNamespace(
        num_draft_tokens=sum(drafts),
        num_draft_tokens_per_req=np.array(drafts),
        query_start_loc_np=np.array(starts),
        req_ids=["a-cond-x", "a-uncond-y", "b-cond-x", "b-uncond-y"],
        num_reqs=4,
    )


def test_mixed_prefill_and_verification(monkeypatch):
    monkeypatch.setenv("VLLM_AUDEX_INVARIANT_ROUTER", "1")
    with verification_groups(batch([0, 7, 12, 16, 20], [0, 0, 3, 3])):
        current = groups()
        assert len(current) == 16
        assert all(len(group.tokens) == 1 for group in current[:12])
        assert all(len(group.tokens) == 2 for group in current[12:])


def test_split_verification_pair_fails(monkeypatch):
    monkeypatch.setenv("VLLM_AUDEX_INVARIANT_ROUTER", "1")
    with pytest.raises(RuntimeError, match="unequal draft lengths"):
        with verification_groups(batch([0, 4, 8, 12, 16], [0, 0, 3, 0])):
            pass
