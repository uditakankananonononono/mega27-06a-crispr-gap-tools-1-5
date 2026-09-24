import json


def test_pyarrow_bench():
    d = json.load(open("results/pyarrow_cache_bench.json"))
    for name, r in d.items():
        assert r["roundtrip_identical"] is True
        assert r["parquet_mb"] < r["csv_mb"]
