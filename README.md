# ToolDefer

Tool-call verification pilot using `mlx-community/Qwen2.5-1.5B-Instruct-4bit` on Apple Silicon. The experiment tests whether assistant attribution and tool-success cues change approval of the same correct and incorrect calls.

The October 5, 2026 run covers the first 300 BFCL `simple_python` items, with six conditions and both answer orders. The prepared dataset contains 598 items; the stored pilot does not evaluate all of them. Raw scores, generation checks, analysis CSVs and figures are included under `results/tooldefer/`.

## Setup and reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest code/tooldefer/tests -q
python code/tooldefer/build_bfcl_pairs.py
python code/tooldefer/tooldefer_run_local.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --answer-format yesno --max-rows 300
python code/tooldefer/tooldefer_analyze.py --input results/tooldefer/bfcl_Qwen2.5-1.5B-Instruct-4bit_yesno_results.jsonl
python code/tooldefer/tooldefer_plots.py --input results/tooldefer/bfcl_Qwen2.5-1.5B-Instruct-4bit_yesno_results.jsonl
```

MLX inference requires Apple Silicon. Existing output files are resumed; use `--output` with a fresh path for a new inference run.

See [the pilot results summary](reports/pilot_results.md) and the [overview figure](results/tooldefer/figures/fig0_overview.png).

## Data and code provenance

This experiment extends [DialDefer](https://github.com/LadyPary/DialDefer). Shared utilities in `code/common/` are retained from that project. Raw benchmark inputs come from the Berkeley Function Calling Leaderboard (BFCL); dataset rights remain with their original providers.
