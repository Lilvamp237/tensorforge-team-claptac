# SafePassage LK

Detects fraudulent foreign-job, student-visa and migration offers aimed at Sri Lankans, and flags when scam tactics shift.
See [docs/PROBLEM_SCOPE.md](docs/PROBLEM_SCOPE.md) for the full problem scope.

> Work in progress: TensorForge 2.0 Final-Forge build (17 Oct 2026).

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/train_baseline.py
PYTHONPATH=src uvicorn safepassage.api:app --reload
```

Or with Docker: `docker build -t safepassage . && docker run -p 8000:8000 safepassage`

## Tests

```bash
pytest
```

## Sections to complete

- Problem and users
- Data sources and licences
- Architecture diagram
- Evaluation results (held-out test set)
- Monitoring and drift response
- Impact statement
- AI tools used: Claude Code (Anthropic) for scaffolding and research
