# Application evaluation tools

Install evaluation dependencies with `python -m pip install -r benchmarks/requirements.txt`
after installing the application's root `requirements.txt`.

These scripts cover separate evaluations of the literature assistant:

- `human_statistics.py`: recompute statistics and bootstrap intervals from the
  18-answer, 12-reviewer generation study.
- `operating_cost.py`: profile online QA latency, token usage and API cost.
  This is an operational profile, not a relevance benchmark.
- `operating_artifacts.py`: validate and format saved operational measurements.
- `grounding_audit.py` and `grounding_artifacts.py`: audit saved online answers
  against their evidence. Automated assessments are distinct from independent
  human generation ratings.
- `trec_capacity_probe.py`: measure local indexing capacity for TREC segments.
- `scalability.py`: export, build, verify and summarize nested offline indexes;
  it does not run a query workload or evaluate relevance.
- `reproducibility_bundle.py`: package source and text-free evaluation artifacts.
- `recompute_supplement.py`: independently recompute human-rating statistics.

Public retrieval evaluations using released LitSearch and TREC qrels live in
[`../public_retrieval/`](../public_retrieval/README.md).

Results remain in their respective directories under `output/benchmarks/`.
Operational profiling and grounding auditing can call paid APIs; recomputation
of saved human scores and public retrieval metrics does not require LLM calls.
