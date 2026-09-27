#!/usr/bin/env python3
"""Self-tests for scripts/batch_sampler.py (no pytest required).

    .venv/bin/python scripts/test_batch_sampler.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_sampler import (plan_batches, plan_epoch, plan_random,  # noqa: E402
                           lengths_from_hist)


def padded(lengths, batch):
    return max(lengths[i] for i in batch) * len(batch)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def test_covers_every_item_once():
    lengths = [32, 40, 8192, 50, 64, 70, 3000, 120, 150, 200, 25, 28]
    plan, _ = plan_batches(lengths, 4, 512, 4, random.Random(1))
    idx = sorted(i for b in plan for i in b)
    check(idx == list(range(len(lengths))), "모든 항목이 정확히 1회 포함")


def test_respects_token_budget_when_possible():
    lengths = [100] * 50 + [110] * 50 + [120] * 50
    plan, st = plan_batches(lengths, 8, 900, 4, random.Random(2))
    check(st["batches_over_token_budget"] == 0, "예산 초과(회피 가능) 배치 0")
    check(all(padded(lengths, b) <= 900 for b in plan), "모든 배치가 900 padded 토큰 이하")


def test_single_item_longer_than_budget_is_flagged_not_hidden():
    lengths = [8192, 10, 10]
    plan, st = plan_batches(lengths, 2, 512, 4, random.Random(3))
    check(st["single_item_over_budget"] == 1, "예산보다 긴 단일 항목은 별도 집계")
    check(st["batches_over_token_budget"] == 0, "회피 가능한 초과로는 세지 않음")


def test_padding_waste_drops_vs_random():
    rng = random.Random(7)
    lengths = [rng.choice([64, 96, 128, 256, 512, 4096]) for _ in range(4000)]
    _, st_rand = plan_random(lengths, 8, 16384, random.Random(11))
    _, st_len = plan_batches(lengths, 8, 16384, 64, random.Random(11))
    check(st_len["padded_tokens"] < st_rand["padded_tokens"],
          f"길이 버킷 패딩({st_len['padded_tokens']:,}) < 랜덤({st_rand['padded_tokens']:,})")
    check(st_len["waste_pct"] <= st_rand["waste_pct"], "낭비율이 증가하지 않음")


def test_bucket_shuffle_changes_order_across_epochs():
    lengths = list(range(1, 200))
    p1, _ = plan_epoch(lengths, 4, None, "length", 4, seed=1)
    p2, _ = plan_epoch(lengths, 4, None, "length", 4, seed=2)
    check(p1 != p2, "에폭마다 배치 구성이 달라짐(버킷 셔플)")
    flat1 = sorted(i for b in p1 for i in b)
    check(flat1 == list(range(len(lengths))), "에폭 내 항목 중복/누락 없음")


def test_batch_size_never_exceeds_micro_batch():
    lengths = [50] * 100
    plan, _ = plan_batches(lengths, 7, None, 8, random.Random(5))
    check(all(len(b) <= 7 for b in plan), "배치 크기가 MICRO_BATCH 를 넘지 않음")


def test_random_mode_matches_legacy_shape():
    lengths = list(range(1, 101))
    plan, st = plan_random(lengths, 8, None, random.Random(0))
    check(len(plan) == (100 + 7) // 8, "random 모드는 기존 고정 크기 청킹과 동일한 배치 수")


def test_hist_expansion_roundtrip():
    hist = {"buckets": [{"upper": 128, "count": 10, "real_tokens": 900, "padded_tokens": 1280,
                         "pad_waste_pct": 0.0},
                        {"upper": 256, "count": 5, "real_tokens": 1000, "padded_tokens": 1280,
                         "pad_waste_pct": 0.0}]}
    import json
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(hist, fh)
        p = fh.name
    got = lengths_from_hist(p)
    check(len(got) == 15, "히스토그램 전개 개수 일치")
    check(all(1 <= x <= 128 for x in got[:10]) and all(129 <= x <= 256 for x in got[10:]),
          "버킷 경계 안에서 값 생성")


def test_collate_shapes_follow_the_plan():
    """End-to-end data check: the real collate must pad to the planned batch max."""
    from train_ddp import collate_train_batch
    items = []
    for n in (20, 64, 130, 131, 400, 25, 33, 900):
        items.append({"ids": list(range(n)), "markers": [1, 2], "target": [0.5, 0.5],
                      "qtype": 0, "label": 0})
    lengths = [len(it["ids"]) for it in items]
    plan, st = plan_batches(lengths, 4, 1200, 4, random.Random(3))
    seen = 0
    for b in plan:
        batch = collate_train_batch([items[i] for i in b], 0)
        want = max(lengths[i] for i in b)
        check(batch["input_ids"].shape == (len(b), want),
              f"배치 shape=(행 {len(b)}, 열 {want}) 패딩 일치")
        check(int(batch["attention_mask"].sum()) == sum(lengths[i] for i in b),
              "attention_mask 합 = 실제 토큰 수")
        seen += len(b)
    check(seen == len(items), "플랜→콜레이트 왕복에서 항목 누락 없음")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} tests")
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    print("ALL TESTS PASSED")
