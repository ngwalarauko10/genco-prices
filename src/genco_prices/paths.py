"""Locations and pinned versions. All paths are relative to the repository root."""

import os
from pathlib import Path

DATASET = "gridfm/opf_small_case118_ieee"
DATASET_REV = "5377c37bd0e75980590788ffdcb9100d3bc330ff"
MODEL = "gridfm/genco-opf-datakit"
MODEL_REV = "2d2401572f6a2d0c1be79841b2def99cf45e1639"
CHECKPOINT = Path("checkpoints/genco-opf-datakit/case118_ieee/small/seed0")
SPLIT_FILES = {  # the split that each checkpoint saved at training time
    "small": "mlflow/train/case118_ieee/small/seed0/465594abc8eb4c96b908363afc84a0e1/artifacts/stats/case118_ieee_scenario_splits.json",
    "base": "mlflow/train/case118_ieee/base/seed0/7bf3bd96b51747ecadb7963af90cc506/artifacts/stats/case118_ieee_scenario_splits.json",
}

RAW = Path("data/opf/case118_ieee/raw")  # datakit tables (bus_data, gen_data, branch_data)
# output of `gridfm_graphkit evaluate --save_output`; GENCO_PRED selects another model (e.g. the Base checkpoint)
PRED = Path(os.environ.get("GENCO_PRED", "results/genco_opf_case118"))
LABELS = Path("results/labels")  # AC-OPF and DC-OPF prices, decoded prices
FIGURES = Path("figures")

BASE_MVA = 100.0
SCEN_PER_PARTITION = 200  # datakit writes scenarios 200*p .. 200*p+199 to scenario_partition=p
