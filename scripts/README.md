# Scripts

| Script | Owner | Does |
|---|---|---|
| `make_dataset.py` | A | EMSCAD + synthetic -> `data/processed/{train,test,shifted}.csv` (columns: text, label, category) |
| `generate_synthetic.py` | A | LLM-generated SL offers in EN/SI/Singlish -> `data/synthetic/` |
| `scrape_slbfe.py` | A | SLBFE job bank -> `data/processed/registry.csv` (personal details removed, polite crawl) |
| `train_baseline.py` | A | TF-IDF + LR -> `models/baseline.joblib` |
| `evaluate.py` | A | Held-out + shifted metrics -> `reports/eval.json` |
| `calibrate_drift.py` | A | Fits the kNN novelty index, calibrates drift thresholds on held-out vs shifted windows -> `models/{novelty.joblib,drift_reference.json}`, `reports/drift_calibration.json` |
| `collect_ikman.py` | B | Public ikman.lk job ads -> `service.scan_text(source="collector")`. `PYTHONPATH=src python scripts/collect_ikman.py --once --max-ads 50` (add `--category jobs` for local jobs, `--interval-min 30` to repeat, `--api http://localhost:8000` to go through the REST API, `--fixtures tests/fixtures/ikman` for an offline demo). Honours robots.txt (incl. wildcards), >=1.5 s between requests, stops on 403/429; strips phones, emails, NICs and contact names before scanning; dedupes via `logs/collector_seen.json`; raw HTML only with `--cache-raw` (gitignored `data/raw/ikman/`) |
| `replay_shifted.py` | B | Streams the shifted set through the service to fire the drift alert in the demo |
