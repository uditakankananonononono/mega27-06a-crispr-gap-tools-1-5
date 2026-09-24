"""Gap 1 extension: 6-dataset CNN cross-dataset matrix (adds sgDesigner,
Hiranniramol 2020 Bioinformatics, plasmid-library assay; 1,309 guides,
57% of scores saturated at the 100% ceiling - reported as a ceiling-tie
artifact case). Same harness/config as the 5-dataset matrix; one seed per
invocation."""
import json
import sys

import numpy as np
import torch

from crisprgap.crossdata import run_cross_dataset, summarize
from crisprgap.data.datasets import (load_doench_fcres, load_doench_v1,
                                     load_deephf, load_deepspcas9,
                                     load_crispron, load_sgdesigner)
import sys as _sys
_sys.path.insert(0, "scripts")
from run_crossdata_cnn import cnn_model  # noqa: E402

torch.set_num_threads(2)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(seed)
    ds = {"fcres": load_doench_fcres(), "v1": load_doench_v1(),
          "deephf_wt": load_deephf("wt", max_n=12000),
          "deepspcas9": load_deepspcas9(max_n=12000),
          "crispron": load_crispron(max_n=12000),
          "sgdesigner": load_sgdesigner()}
    res = run_cross_dataset(ds, {"cnn": cnn_model()}, seeds=(seed,))
    out = {"raw": {str(k): v for k, v in res.items()}, "summary": summarize(res)}
    json.dump(out, open(f"results/crossdata_cnn_6ds_seed{seed}.json", "w"), indent=2)
    print(json.dumps(out["summary"]["cnn"]["sgdesigner"], indent=2))
