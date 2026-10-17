Knowledge base for the RAG claim checker: short hand-written notes summarising official public rules, one file per topic, each with its source URL.

- `slbfe_registry.csv`: 3,068 agencies (1,010 licensed, 105 suspended, 1,953 cancelled) from SLBFE's public agency lists. Name, licence number and status only.
- `slbfe_job_orders.csv`: 32,686 SLBFE-approved job orders from 540 agencies' public Online Job Bank pages, with the maximum fee each agency may charge.
- Both were built on 17 Oct 2026 by `scripts/scrape_slbfe.py --jobbank`. Addresses, phone and fax numbers and emails on those pages are never parsed.
