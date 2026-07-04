from __future__ import annotations

from scripts.evaluate_db_agent import select_sample_rows, shard_eval_rows, stable_query_shard


def _rows(count: int) -> list[dict[str, str]]:
    return [
        {
            "dataset_name": "auto",
            "split_name": "test",
            "qid": f"q{i:03d}",
        }
        for i in range(count)
    ]


def test_shard_eval_rows_covers_all_rows_without_overlap() -> None:
    rows = _rows(101)

    shards = [shard_eval_rows(rows, shard_count=4, shard_index=index) for index in range(4)]
    merged_keys = [
        (row["dataset_name"], row["split_name"], row["qid"])
        for shard in shards
        for row in shard
    ]

    assert len(merged_keys) == len(rows)
    assert len(set(merged_keys)) == len(rows)
    assert set(merged_keys) == {(row["dataset_name"], row["split_name"], row["qid"]) for row in rows}


def test_stable_query_shard_is_deterministic() -> None:
    row = {"dataset_name": "real", "split_name": "test", "qid": "42"}

    assert stable_query_shard(row, shard_count=4) == stable_query_shard(dict(row), shard_count=4)


def test_proportional_sample_keeps_auto_heavy_distribution() -> None:
    rows = [
        {"dataset_name": "AutoScholarQuery", "split_name": "test", "qid": f"a{i:04d}"}
        for i in range(1000)
    ] + [
        {"dataset_name": "RealScholarQuery", "split_name": "test", "qid": f"r{i:04d}"}
        for i in range(50)
    ]

    sample = select_sample_rows(rows, max_total_queries=100, sample_profile="proportional")
    counts = {}
    for row in sample:
        counts[row["dataset_name"]] = counts.get(row["dataset_name"], 0) + 1

    assert len(sample) == 100
    assert counts["AutoScholarQuery"] == 95
    assert counts["RealScholarQuery"] == 5


def test_balanced_sample_keeps_real_queries_visible() -> None:
    rows = [
        {"dataset_name": "AutoScholarQuery", "split_name": "test", "qid": f"a{i:04d}"}
        for i in range(1000)
    ] + [
        {"dataset_name": "RealScholarQuery", "split_name": "test", "qid": f"r{i:04d}"}
        for i in range(50)
    ]

    sample = select_sample_rows(rows, max_total_queries=20, sample_profile="balanced")
    counts = {}
    for row in sample:
        counts[row["dataset_name"]] = counts.get(row["dataset_name"], 0) + 1

    assert len(sample) == 20
    assert counts["AutoScholarQuery"] == 10
    assert counts["RealScholarQuery"] == 10
