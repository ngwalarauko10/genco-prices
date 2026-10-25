#!/usr/bin/env bash
# Every number and figure of the post, from the raw data. About 9 GB of disk; 2-3 h on a laptop.
# Steps 1-3 and 8 need the `genco` dependency group (torch, gridfm-graphkit); the other steps need only the core group.
# Without steps 1-6: results/per_scenario_test3000.csv and results/headline_test3000.csv hold the evaluated numbers.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # one thread per process; the LPs are small

# 1. Data and model, at the revisions we used
uv run hf download gridfm/opf_small_case118_ieee --repo-type dataset --revision 5377c37bd0e75980590788ffdcb9100d3bc330ff \
    --exclude "solver_log/*" --exclude "*.log" --local-dir data/opf/case118_ieee/raw
uv run hf download gridfm/genco-opf-datakit --revision 2d2401572f6a2d0c1be79841b2def99cf45e1639 \
    --include "case118_ieee/small/seed0/*" --local-dir checkpoints/genco-opf-datakit

# 2. GENCO predictions on the test partition (21,199 scenarios; GPU optional)
CK=checkpoints/genco-opf-datakit/case118_ieee/small/seed0
MLFLOW_ALLOW_FILE_STORE=true uv run --group genco gridfm_graphkit evaluate --config_conversion auto_inline \
    --config $CK/case118_ieee_small_seed0.yaml --data_path data/opf --model_path $CK/last.pt \
    --normalizer_stats $CK/normalizer_stats.pt --log_dir logs/mlflow_opf_eval --num_workers 0 \
    --save_output --output_path results/genco_opf_case118

# 3. The evaluated scenarios must be the checkpoint's saved test partition
uv run python -m genco_prices.check_split

# 4. AC-OPF and DC-OPF prices for the 3,000 sampled test scenarios (results/test_ids_3000.txt, seed 0)
uv run python -m genco_prices.labels --scenarios results/test_ids_3000.txt --workers 1 --tag test3000

# 5. Tune the two decoder tolerances on one third of the load scenarios, decode, report on the other two thirds
uv run python -m genco_prices.evaluate tune --tag test3000
uv run python -m genco_prices.classify --tag test3000
uv run python -m genco_prices.evaluate decode --tag test3000
uv run python -m genco_prices.evaluate report --tag test3000

# 6. Figures
uv run python -m genco_prices.figures --tag test3000

# 7. Optional (WITH_BASE=1 ./reproduce.sh): a second model, GENCO Base. Its saved split equals Small's,
#    so the same labels apply.
if [ "${WITH_BASE:-0}" = 1 ]; then
    uv run hf download gridfm/genco-opf-datakit --revision 2d2401572f6a2d0c1be79841b2def99cf45e1639 \
        --include "case118_ieee/base/seed0/*" --local-dir checkpoints/genco-opf-datakit
    CK=checkpoints/genco-opf-datakit/case118_ieee/base/seed0
    MLFLOW_ALLOW_FILE_STORE=true uv run --group genco gridfm_graphkit evaluate --config_conversion auto_inline \
        --config $CK/case118_ieee_base_seed0.yaml --data_path data/opf --model_path $CK/last.pt \
        --normalizer_stats $CK/normalizer_stats.pt --log_dir logs/mlflow_opf_eval_base --num_workers 0 \
        --save_output --output_path results/genco_opf_case118_base
    for t in summary lmp branch; do ln -sf case118_test3000_$t.parquet results/labels/case118_base_$t.parquet; done
    export GENCO_PRED=results/genco_opf_case118_base
    uv run python -m genco_prices.check_split --model base
    uv run python -m genco_prices.evaluate tune --tag base
    uv run python -m genco_prices.classify --tag base
    uv run python -m genco_prices.evaluate decode --tag base
    uv run python -m genco_prices.evaluate report --tag base
    unset GENCO_PRED
    uv run python -m genco_prices.evaluate compare --tag test3000 --other base
fi

# 8. Optional (WITH_PROBE=1 ./reproduce.sh; GPU recommended): probes on GENCO's final embeddings. Needs steps 1, 2, 5.
#    Embeddings of the 3,000 labelled scenarios (about 1 min on a laptop GPU), then every probe row in one run
#    (about 20 min, one thread, about 2 GB of memory).
if [ "${WITH_PROBE:-0}" = 1 ]; then
    CK=checkpoints/genco-opf-datakit/case118_ieee/small/seed0
    MLFLOW_ALLOW_FILE_STORE=true uv run --group genco python scripts/predict_embeddings.py \
        --keep results/test_ids_3000.txt -- predict --config_conversion auto_inline \
        --config $CK/case118_ieee_small_seed0.yaml --data_path data/opf --model_path $CK/last.pt \
        --normalizer_stats $CK/normalizer_stats.pt --log_dir logs/mlflow_opf_predict_emb --num_workers 0 \
        --get_embeddings --output_path results/genco_opf_case118_emb
    uv run python -m genco_prices.probe fit --tag test3000
    uv run python -m genco_prices.probe figure --tag test3000
fi

# 9. Optional (WITH_MARKET=1 ./reproduce.sh): real data, the Core day-ahead market, 02-15/10/2026. Downloads JAO tables,
#    the final domain of 14-15/10 and Energy-Charts prices into data/ (rate limits: about 30 min); analysis about 3 min.
if [ "${WITH_MARKET:-0}" = 1 ]; then
    uv run python -m genco_prices.market fetch
    uv run python -m genco_prices.market analyse    # results/market_core_*, figures/fig3_core_duals.png
fi
