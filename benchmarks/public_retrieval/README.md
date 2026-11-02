# Public retrieval comparison (2026-10-29)

Authorized scope: full LitSearch (64,183 papers / 597 queries) and TREC RAG 2024
reduced corpus (50 topics: all 27 fully manual topics plus 23 seeded additional
UMBRELA topics). Application source, production Chroma indexes and manuscript
are not modified. No generation or LLM-judging API is called.

Outputs: `output/benchmarks/public_retrieval_20261029/{litsearch,trec_rag_2024}`.

## Run / resume

### Offline RRF sensitivity

```powershell
python benchmarks/public_retrieval/rrf_sensitivity.py
```

This re-fuses the saved plain BM25/dense top-100 lists over
`k = [10,20,40,60,80,90]` and BM25 weights `alpha = [0,.25,.5,.75,1]`.
No retrieval, embeddings, LLM/API calls, queries or relevance labels are generated.
The original rankings, qrels, application defaults and headline scores remain fixed.

Outputs are separate: `output/benchmarks/rrf_sensitivity_20261102/`.
`REPORT.md`, `summary.json`, grid/per-query CSVs and heatmaps report the
author-written LitSearch and manual TREC priority subsets, full LitSearch,
and separate mixed-label/UMBRELA TREC groups. Checks reproduce every original
default ranking/metric and endpoint ranking and verify input hashes unchanged.

The smoothing grid was revised at the author's request on 2026-11-03 after
the earlier results were seen; all 30 revised cells are retained.
This is **post-hoc sensitivity analysis, not held-out tuning**. Pointwise
95% paired query-bootstrap intervals are exploratory. nDCG contrasts additionally
have centered max-absolute bootstrap simultaneous bands within each dataset/group,
not across groups. The default stays `k=60, alpha=.5`; largest observed means
are in-sample descriptions. The grid concerns plain dense fusion conditional on
the existing candidate depth and similarity floor, not the reranked configuration.

### Original public benchmark

```powershell
python -u benchmarks/public_retrieval/run.py --dataset litsearch
python -u benchmarks/public_retrieval/prepare_trec.py --collect
python -u benchmarks/public_retrieval/run.py --dataset trec_rag_2024
```

Use `--report-only` to recompute scores from saved rankings. Individual query
rankings, embedding batches and TREC shard extraction checkpoints are retained.
Do not change configuration mid-run. Preparation requires public network data.

## Fixed methods

- BM25Okapi, lowercase Unicode word tokenization, title plus text.
- Dense: all-MiniLM-L6-v2 on CUDA, max length 256, normalized vectors,
  Chroma cosine HNSW (construction/search ef 100, neighbors 16).
- Up to 100 candidates each branch; dense similarity floor .1; RRF k=60.
- Fourth method: FlashRank ms-marco-MiniLM-L-12-v2 reranks the dense top 100
  **before** RRF, max length 512, CPU ONNX 4 intra-op / 1 inter-op threads,
  microbatches of 8. No silent fallback if the reranker is unavailable.
- Three warmup queries; each query is measured once. Component-summed timings
  exclude acquisition, model loading, indexing and generation. Build times are
  separate; simultaneous background extraction may affect observed latency.

## Labels and scoring

LitSearch uses original target-paper IDs (binary), not exhaustive human relevance
judgments. Queries include inline-citation synthetic and author-written groups;
report full set, individual source groups and combined author-written subset.
Representation is `title: <title>. abstract: <abstract>`; full text is not indexed.

TREC primary scores use NIST fully manual qrels for the 27 manual topics and
UMBRELA qrels for the other 23. Report these groups separately. The 50-topic
aggregate mixes label sources and must be described as such. Original UMBRELA
qrels are also retained for the manual group, not used to override human labels.

TREC corpus selection retains the union of all available assessed segments for
selected topics, including grade 0. Seeded random distractors raise the total to
64,200 (or more if assessed segments exceed this). Extraction reads the public
`drexalt/msmarco-2.1-segmented` mirror at a recorded revision. Per-shard Bernoulli
sampling followed by a fixed-size random sample selects distractors independently
of grades and method scores. Every required ID must be found before scoring.
This is a **judgment-pool-derived reduced-corpus experiment**, not the official
full-corpus TREC benchmark. It has fewer distractors and possibly easier rankings.

### Approved source-data exception

The extraction scanned all 113,520,750 segments across 60 source shards. One
qrels ID, `msmarco_v2.1_doc_01_23#45_67`, was not found. All its grades in the
selected topics are zero. The user explicitly approved omitting this unavailable
ID from the corpus requirement while **retaining original qrels unchanged**.
17,662 assessed segments plus 46,538 seeded random distractors form the final
64,200-passage corpus. No known relevant passage is omitted by this exception.
The missing ID and approval are recorded in the output manifest. To rebuild
from the retained shard samples with this exact authorized exception:

```powershell
python -u benchmarks/public_retrieval/prepare_trec.py --collect --allow-missing-zero-id 'msmarco_v2.1_doc_01_23#45_67'
```

Macro MRR, Hit, Recall, nDCG at 5/10/20/100. Binary relevance threshold >=1;
nDCG gain 2^grade-1. Explicit exponential gain may differ from official evaluator
defaults, so these scores are not asserted comparable with official tables.
Unjudged results score zero conventionally but are not verified negatives.
Recall is against known qrels, not all relevant corpus items. Empty relevant sets
score zero and remain in the evaluation. Confidence intervals use 10,000 paired
query bootstrap draws, seed 20261029, unadjusted for multiplicity. Queries may be
dependent (e.g. shared target papers); intervals are exploratory.

The reporting script additionally emits linear-gain nDCG (compatible with the
usual trec_eval gain convention), TREC binary-threshold >=2 sensitivity metrics,
and standard TREC-format run/qrels files. Exported run scores are 1/rank to
preserve ranking order; they are not original retrieval scores. The accompanying
plots and main table use explicitly documented exponential-gain nDCG.

## Sources

- https://github.com/princeton-nlp/LitSearch
- https://huggingface.co/datasets/princeton-nlp/LitSearch
- https://trec.nist.gov/data/rag2024.html
- https://trec-rag.github.io/annoucements/umbrela-qrels/

Full source URLs, retrieval data revisions, corpus hashes, versions and machine
information are saved with the results. No superiority is assumed before scoring.
