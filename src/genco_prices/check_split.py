"""Check that the evaluated scenarios are the checkpoint's own test partition (no training scenarios).

`gridfm_graphkit evaluate` rebuilds the split from the data and a seed; it does not read it from the checkpoint.
This script compares the result with the split file that the model repository saved at training time.

    uv run python -m genco_prices.check_split                       # GENCO Small
    GENCO_PRED=results/genco_opf_case118_base uv run python -m genco_prices.check_split --model base
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .paths import MODEL, MODEL_REV, PRED, SPLIT_FILES


def saved_split(model: str) -> dict[str, set[int]]:
    d = json.loads(open(hf_hub_download(MODEL, SPLIT_FILES[model], revision=MODEL_REV)).read())
    return {k: set(v) for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SPLIT_FILES), default="small")
    model = ap.parse_args().model
    split = saved_split(model)
    test, train = split["test"], split["train"] | split["val"]
    order = pd.read_parquet(PRED / "case118_ieee_gen_predictions.parquet", columns=["scenario"]).scenario.unique()
    ours = set(order)
    sample = set(np.loadtxt("results/test_ids_3000.txt", dtype=int))
    print(f"{model} saved split: {len(split['train'])} train, {len(split['val'])} val, {len(test)} test")
    print(f"evaluated: {len(ours)}; equal to saved test partition: {ours == test}; overlap with train/val: {len(ours & train)}")
    print(f"labelled sample: {len(sample)}; inside saved test partition: {sample <= test}")
    assert ours == test and sample <= test
    if model == "small":
        redraw = set(np.random.default_rng(0).choice(order, 3000, replace=False))  # how the sample was drawn
        print(f"sample equal to the seed-0 draw: {sample == redraw}")
        assert sample == redraw
    else:  # the labels of the Small run are reused: the two saved splits must be identical
        same = split == saved_split("small")
        print(f"saved split identical to Small's (train, val, test): {same}")
        assert same


if __name__ == "__main__":
    main()
