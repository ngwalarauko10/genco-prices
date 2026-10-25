"""Export GENCO's final embeddings for the labelled test scenarios (the input of genco_prices.probe).

`gridfm_graphkit predict --get_embeddings` runs GENCO on the checkpoint's test partition (21,199 scenarios). It writes
the bus and generator predictions and the final hidden embeddings: 192 values per bus and per generator. For all test
scenarios, the embeddings need about 2.8 GB as float32, and the CLI keeps several copies in memory. On a 16 GB laptop
this can stop the machine. The probe needs only the 3,000 labelled scenarios (results/test_ids_3000.txt).

What the wrapper patches: the predict step of graphkit calls `embedding_table_from_tensor` (imported into
gridfm_graphkit.tasks.opf_task) once per batch, for the bus and the generator embeddings. The wrapper replaces this name
in opf_task only. The replacement keeps the rows of the scenarios in --keep, on the device before the copy to the CPU,
and then calls the original function. The model, the batches and the bus and generator prediction tables do not
change. The wrapper also writes seen_scenarios.txt: the ids of all scenarios that the model saw, to check the split.

Version: tested with gridfm-graphkit at commit 8a8defb (8a8defbc9ab41caa65b07c75e8e1ee23636877cd), the pin of the
`genco` dependency group in pyproject.toml. Another commit can change the patched function or its call; the wrapper
stops if the function is missing and warns if the commit is different.

Run from any directory; all paths are relative to the repository root (the wrapper changes to it first). Steps 1-2 of
reproduce.sh give the checkpoint and the data. Same flags as reproduce.sh step 2, with `predict --get_embeddings`:

    CK=checkpoints/genco-opf-datakit/case118_ieee/small/seed0
    MLFLOW_ALLOW_FILE_STORE=true uv run --group genco python scripts/predict_embeddings.py \
        --keep results/test_ids_3000.txt -- predict --config_conversion auto_inline \
        --config $CK/case118_ieee_small_seed0.yaml --data_path data/opf --model_path $CK/last.pt \
        --normalizer_stats $CK/normalizer_stats.pt --log_dir logs/mlflow_opf_predict_emb --num_workers 0 \
        --get_embeddings --output_path results/genco_opf_case118_emb

GPU: recommended. Our run (25/10/2026) took about 1 min for the 21,199 scenarios on a laptop GPU with 8 GB of
memory. A CPU run also works but takes much longer (not measured). Check: the dispatch of our run is the same as the
dispatch of `gridfm_graphkit evaluate` (reproduce.sh step 2) to within 0.052 MW (largest generator difference over the
3,000 labelled scenarios).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from gridfm_graphkit.__main__ import main
from gridfm_graphkit.tasks import opf_task

ROOT = Path(__file__).resolve().parents[1]
GRAPHKIT_COMMIT = "8a8defbc9ab41caa65b07c75e8e1ee23636877cd"


def check_graphkit() -> None:
    if not hasattr(opf_task, "embedding_table_from_tensor"):
        sys.exit("gridfm_graphkit.tasks.opf_task has no embedding_table_from_tensor: use graphkit commit 8a8defb")
    try:
        commit = json.loads(metadata.distribution("gridfm-graphkit").read_text("direct_url.json") or "{}")
        commit = commit.get("vcs_info", {}).get("commit_id")
    except metadata.PackageNotFoundError:
        commit = None
    if commit != GRAPHKIT_COMMIT:
        warnings.warn(f"gridfm-graphkit commit {commit}, tested with {GRAPHKIT_COMMIT[:7]}", stacklevel=2)


def run() -> None:
    os.chdir(ROOT)
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: predict_embeddings.py [--keep IDS] -- predict ... --get_embeddings --output_path DIR")
    sep = argv.index("--")
    ap = argparse.ArgumentParser(description="graphkit predict that keeps the embeddings of some scenarios only")
    ap.add_argument("--keep", default="results/test_ids_3000.txt", help="scenario ids, one per line")
    own, rest = ap.parse_args(argv[:sep]), argv[sep + 1:]
    if rest[:1] != ["predict"] or "--get_embeddings" not in rest or "--output_path" not in rest:
        sys.exit("after --: graphkit `predict` with --get_embeddings and --output_path")
    check_graphkit()
    keep = np.array(sorted(int(x) for x in Path(own.keep).read_text().split()))
    out_dir = Path(rest[rest.index("--output_path") + 1])
    seen: set[int] = set()
    original = opf_task.embedding_table_from_tensor

    def filtered(embedding: torch.Tensor, *, id_columns: dict[str, np.ndarray], prefix: str = "emb") -> dict:
        scen = id_columns["scenario"]
        seen.update(int(s) for s in np.unique(scen))
        mask = np.isin(scen, keep)
        rows = torch.from_numpy(np.flatnonzero(mask)).to(embedding.device)
        return original(embedding.index_select(0, rows), id_columns={k: v[mask] for k, v in id_columns.items()},
                        prefix=prefix)

    opf_task.embedding_table_from_tensor = filtered
    sys.argv = [sys.argv[0]] + rest
    main()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seen_scenarios.txt").write_text("\n".join(str(s) for s in sorted(seen)) + "\n")
    print(f"embeddings kept for {np.isin(keep, sorted(seen)).sum()} of {len(keep)} scenarios; model saw {len(seen)}")


if __name__ == "__main__":
    run()
