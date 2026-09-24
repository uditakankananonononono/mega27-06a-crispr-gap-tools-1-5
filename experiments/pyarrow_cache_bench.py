"""pyarrow parquet caching benchmark: CSV vs parquet load time and size for
the repo's largest curated tables (crisprSQL, DepMap guide efficacy,
PRIDICT v1). Parquet cache written to data/cache/ (gitignored); loaders
keep CSV as source of truth - this is an engineering artifact.
"""
import json
import os
import time

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

TABLES = {
    "crisprsql_100720": "data/crisprsql/100720.csv",
    "depmap_guide_efficacy": "data/depmap/CRISPRInferredGuideEfficacy.csv",
}
os.makedirs("data/cache", exist_ok=True)
out = {}
for name, path in TABLES.items():
    t0 = time.time()
    df = pd.read_csv(path, low_memory=False)
    csv_t = time.time() - t0
    pq_path = f"data/cache/{name}.parquet"
    pq.write_table(pa.Table.from_pandas(df), pq_path, compression="zstd")
    t0 = time.time()
    df2 = pd.read_parquet(pq_path)
    pq_t = time.time() - t0
    assert df.equals(df2), name
    out[name] = {
        "rows": int(len(df)),
        "csv_mb": round(os.path.getsize(path) / 1e6, 1),
        "parquet_mb": round(os.path.getsize(pq_path) / 1e6, 1),
        "csv_load_s": round(csv_t, 3),
        "parquet_load_s": round(pq_t, 3),
        "roundtrip_identical": True,
    }
json.dump(out, open("results/pyarrow_cache_bench.json", "w"), indent=2)
print(json.dumps(out, indent=2))
