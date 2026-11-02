"""Offline, post-hoc RRF sensitivity analysis of frozen public retrieval runs.

No model imports, network requests, retrieval, generation, or new relevance labels.
Run from the repository root: python benchmarks/public_retrieval/rrf_sensitivity.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.public_retrieval.run import metrics


SOURCE = ROOT / "output/benchmarks/public_retrieval_20261029"
OUTPUT = ROOT / "output/benchmarks/rrf_sensitivity_20261102"
K_VALUES = (10, 20, 40, 60, 80, 90)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONFIGS = tuple((k, a) for k in K_VALUES for a in ALPHAS)
BASELINE = CONFIGS.index((60, 0.5))
METRICS = ("nDCG@10", "MRR@10", "Recall@10", "Hit@10", "nDCG_linear@10")
SEED = 20261102
BOOTSTRAPS = 10000


def weighted_rrf(lexical, dense, k=60, alpha=0.5, cutoff=100):
    """Fuse unique branch IDs; missing membership and zero weight contribute zero.

    Sort tied scores by document ID, matching the original public-run convention.
    Multiplying both original equal branch weights by 0.5 preserves the ranking.
    """
    if not math.isfinite(k) or k < 0:
        raise ValueError("k must be finite and nonnegative")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    if cutoff < 0:
        raise ValueError("cutoff must be nonnegative")
    scores = {}
    for ranking, weight in ((lexical, alpha), (dense, 1 - alpha)):
        if len(ranking) != len(set(ranking)):
            raise ValueError("Duplicate candidate IDs in a branch")
        if weight == 0:
            continue
        for rank, docid in enumerate(ranking, 1):
            scores[docid] = scores.get(docid, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))[:cutoff]


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_means(values, repetitions=BOOTSTRAPS, seed=SEED):
    """Paired query resampling: one shared set of draws for every configuration."""
    n = len(values)
    draws = np.random.default_rng(seed).integers(0, n, size=(repetitions, n))
    counts = np.zeros((repetitions, n), dtype=np.float64)
    np.add.at(counts, (np.arange(repetitions)[:, None], draws), 1)
    return (counts @ values.reshape(n, -1) / n).reshape((repetitions,) + values.shape[1:])


def load_dataset(base, dataset):
    folder = base / dataset
    queries = read(folder / "queries.json")
    ids = [q["id"] for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query IDs")
    hashes = {f"{dataset}/queries.json": digest(folder / "queries.json")}
    manifest = folder / "run_manifest.json"
    hashes[f"{dataset}/run_manifest.json"] = digest(manifest)
    provenance = read(manifest)
    if provenance["rrf_k"] != 60 or provenance["candidates"] != 100:
        raise ValueError("Source configuration differs from the frozen public benchmark")
    values = np.zeros((len(queries), len(CONFIGS), len(METRICS)))
    rankings, diagnostics, per_query = {}, [], []
    saved_metrics = {}
    metrics_path = folder / "metrics_per_query.csv"
    hashes[f"{dataset}/metrics_per_query.csv"] = digest(metrics_path)
    with metrics_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == "hybrid_rrf":
                saved_metrics[row["query_id"]] = row
    for qi, q in enumerate(queries):
        path = folder / "runs" / (q["id"] + ".json")
        hashes[f"{dataset}/runs/{path.name}"] = digest(path)
        run = read(path)
        if run["query_id"] != q["id"] or run["query"] != q["query"]:
            raise ValueError(f"Mismatched query checkpoint: {q['id']}")
        lexical, dense = run["rankings"]["bm25"], run["rankings"]["dense"]
        assert len(lexical) <= 100 and len(dense) <= 100
        baseline = weighted_rrf(lexical, dense)
        assert baseline == run["rankings"]["hybrid_rrf"], q["id"]
        query_rankings = []
        for ci, (k, alpha) in enumerate(CONFIGS):
            ranking = weighted_rrf(lexical, dense, k, alpha)
            scores = metrics(ranking, q["qrels"], 10)
            values[qi, ci] = [scores[m] for m in METRICS]
            if alpha == 0:
                assert ranking == dense
            if alpha == 1:
                assert ranking == lexical
            query_rankings.append({"k": k, "alpha": alpha, "top10": ranking[:10]})
            per_query.append({"dataset": dataset, "query_id": q["id"], "source_group": q["group"],
                "k": k, "alpha": alpha, **{m: scores[m] for m in METRICS},
                "judged_fraction_at_10": sum(d in q["qrels"] for d in ranking[:10]) / 10,
                "top10_changed_from_default": ranking[:10] != baseline[:10]})
        for mi, metric in enumerate(METRICS):
            assert abs(values[qi, BASELINE, mi] - float(saved_metrics[q["id"]][metric])) < 1e-12
        rankings[q["id"]] = query_rankings
        diagnostics.append({"query_id": q["id"], "lexical_candidates": len(lexical),
            "dense_candidates": len(dense), "overlap": len(set(lexical) & set(dense))})
    return queries, values, per_query, rankings, diagnostics, hashes


def group_results(dataset, group, queries, values, selected):
    v = values[selected]
    boot = bootstrap_means(v)
    means = v.mean(axis=0)
    limits = np.quantile(boot, [0.025, 0.975], axis=0)
    diff = means - means[BASELINE]
    boot_diff = boot - boot[:, BASELINE:BASELINE + 1]
    diff_limits = np.quantile(boot_diff, [0.025, 0.975], axis=0)
    # Descriptive simultaneous band for the entire nDCG grid relative to default.
    # The centered max-absolute bootstrap statistic accounts for scanning the grid.
    critical = float(np.quantile(np.max(np.abs(boot_diff[:, :, 0] - diff[None, :, 0]), axis=1), 0.95))
    rows = []
    for ci, (k, alpha) in enumerate(CONFIGS):
        for mi, metric in enumerate(METRICS):
            rows.append({"dataset": dataset, "group": group, "queries": len(selected),
                "k": k, "alpha": alpha, "metric": metric, "mean": float(means[ci, mi]),
                "ci_low": float(limits[0, ci, mi]), "ci_high": float(limits[1, ci, mi]),
                "delta_from_default": float(diff[ci, mi]),
                "delta_ci_low": float(diff_limits[0, ci, mi]),
                "delta_ci_high": float(diff_limits[1, ci, mi]),
                "simultaneous_delta_low": float(diff[ci, mi] - critical) if mi == 0 and ci != BASELINE else (0.0 if mi == 0 else ""),
                "simultaneous_delta_high": float(diff[ci, mi] + critical) if mi == 0 and ci != BASELINE else (0.0 if mi == 0 else "")})
    ndcg = means[:, 0]
    # Largest means are descriptive in-sample observations, never selected models.
    best = int(np.argmax(ndcg))
    k_only = [i for i, (_, a) in enumerate(CONFIGS) if a == 0.5]
    weight_only = [i for i, (k, _) in enumerate(CONFIGS) if k == 60]
    interior = [i for i, (_, a) in enumerate(CONFIGS) if 0 < a < 1]
    summary = {"dataset": dataset, "group": group, "queries": len(selected),
        "default_ndcg": float(ndcg[BASELINE]),
        "k_only_min": float(ndcg[k_only].min()), "k_only_max": float(ndcg[k_only].max()),
        "weight_only_min": float(ndcg[weight_only].min()), "weight_only_max": float(ndcg[weight_only].max()),
        "interior_grid_min": float(ndcg[interior].min()), "interior_grid_max": float(ndcg[interior].max()),
        "largest_observed_k": CONFIGS[best][0], "largest_observed_alpha": CONFIGS[best][1],
        "largest_observed_ndcg": float(ndcg[best]), "largest_observed_delta": float(diff[best, 0]),
        "largest_observed_pointwise_delta_ci": diff_limits[:, best, 0].tolist(),
        "largest_observed_simultaneous_delta_ci": [float(diff[best, 0] - critical), float(diff[best, 0] + critical)] if best != BASELINE else [0.0, 0.0],
        "simultaneous_band_halfwidth": critical,
        "grid_cells_with_positive_simultaneous_delta": sum(float(diff[i, 0]) - critical > 0 for i in range(len(CONFIGS)) if i != BASELINE),
        "role": "post-hoc sensitivity; no held-out tuning or configuration replacement"}
    return rows, summary


def figures(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    panels = [("litsearch", "author_written", "LitSearch: author-written (246)"),
        ("trec_rag_2024", "fully_manual_27", "TREC: manual topics (27), reduced corpus"),
        ("litsearch", "all", "LitSearch: full release (597)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), constrained_layout=True)
    for ax, (dataset, group, title) in zip(axes, panels):
        selected = [r for r in rows if r["dataset"] == dataset and r["group"] == group and r["metric"] == "nDCG@10"]
        a = np.array([[next(r["mean"] for r in selected if r["k"] == k and r["alpha"] == alpha)
            for alpha in ALPHAS] for k in K_VALUES])
        im = ax.imshow(a, cmap="Blues", vmin=0, vmax=0.6, aspect="auto")
        for yi in range(len(K_VALUES)):
            for xi in range(len(ALPHAS)):
                ax.text(xi, yi, f"{a[yi, xi]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if a[yi, xi] > 0.36 else "#17212b")
        ax.add_patch(Rectangle((1.5, 2.5), 1, 1, fill=False, edgecolor="#e27622", linewidth=2.3))
        ax.set_xticks(range(5), ["0", ".25", ".50", ".75", "1"])
        ax.set_yticks(range(len(K_VALUES)), K_VALUES)
        ax.set_xlabel(r"BM25 weight $\alpha$ (dense weight $1-\alpha$)", fontsize=9)
        ax.set_ylabel(r"RRF constant $k$")
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=.82, label="nDCG@10 (exponential gain)")
    fig.suptitle("Post-hoc sensitivity of plain RRF; orange border = unchanged default", fontsize=12)
    fig.savefig(out / "rrf_sensitivity.png", dpi=220)
    fig.savefig(out / "rrf_sensitivity.svg")
    plt.close(fig)

    # A larger type size and just the two priority subsets for the manuscript.
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), constrained_layout=True)
    for ax, (dataset, group, title) in zip(axes, panels[:2]):
        selected = [r for r in rows if r["dataset"] == dataset and r["group"] == group and r["metric"] == "nDCG@10"]
        a = np.array([[next(r["mean"] for r in selected if r["k"] == k and r["alpha"] == alpha)
            for alpha in ALPHAS] for k in K_VALUES])
        ax.imshow(a, cmap="Blues", vmin=0, vmax=.6, aspect="auto")
        for yi in range(len(K_VALUES)):
            for xi in range(len(ALPHAS)):
                ax.text(xi, yi, f"{a[yi, xi]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if a[yi, xi] > .36 else "#17212b")
        ax.add_patch(Rectangle((1.5, 2.5), 1, 1, fill=False, edgecolor="#e27622", linewidth=2))
        ax.set_xticks(range(5), ["0", ".25", ".50", ".75", "1"], fontsize=9)
        ax.set_yticks(range(len(K_VALUES)), K_VALUES, fontsize=9)
        ax.set_xlabel(r"BM25 weight $\alpha$", fontsize=10)
        ax.set_ylabel(r"RRF constant $k$", fontsize=10)
        ax.set_title("LitSearch: author-written (246)" if dataset == "litsearch"
            else "TREC: manual topics (27)", fontsize=10)
    fig.savefig(out / "rrf_sensitivity_primary.png", dpi=240)
    fig.savefig(out / "rrf_sensitivity_primary.svg")
    plt.close(fig)


def reports(out, rows, summaries):
    lines = ["# RRF constant and branch-weight sensitivity", "",
        "This is a post-hoc sensitivity analysis, not held-out tuning. The application",
        "and the original public benchmark results retain k=60 and equal weights.", "",
        "## Protocol", "",
        "Score(d) = alpha/(k + lexical_rank(d)) + (1-alpha)/(k + dense_rank(d)).",
        "Only a branch containing d contributes; zero-weight branches contribute nothing.",
        "k = " + ", ".join(map(str, K_VALUES)) + "; alpha = 0, .25, .5, .75, 1.",
        "The author requested this revised grid on 2026-11-03 after the original",
        "evaluation and previous sensitivity results had been seen; it was fixed",
        "before rerunning. It remains a retrospective analysis, not a new test set.",
        f"All {len(CONFIGS)} grid cells are retained, including the repeated single-branch endpoints",
        f"({len(K_VALUES) * (len(ALPHAS) - 2) + 2} distinct configurations). Lexical/dense top-100 candidate rankings, original",
        "released qrels, the cosine floor, metric definitions and ID tie-breaking are fixed.",
        "The analysis concerns plain dense RRF; it does not analyze the reranked branch.",
        "No new retrieval, embedding, LLM generation, judging, query or label is used.", "",
        "## Uncertainty", "",
        "10,000 paired query-bootstrap resamples (seed 20261102) provide pointwise",
        "percentile 95% intervals. For nDCG contrasts against the original default,",
        "a centered max-absolute bootstrap statistic additionally gives a simultaneous",
        "band over the complete grid within each dataset/group. These are approximate",
        "exploratory bands, not corrections across datasets/groups or held-out claims.",
        "For each group, critical = quantile_0.95(max_config |delta* - delta_hat|).",
        "Each non-default contrast is reported as delta_hat +/- critical. The default",
        "contrast is identically zero. Shared target papers may induce query dependence.", "",
        "## Summary: nDCG@10", "",
        "| Dataset / group | n | Default | k sweep, alpha=.5 | Weight sweep, k=60 | Largest observed (k, alpha) | Mean |",
        "|---|---:|---:|---|---|---|---:|"]
    for s in summaries:
        lines.append(f"| {s['dataset']} / {s['group']} | {s['queries']} | {s['default_ndcg']:.4f} | "
            f"{s['k_only_min']:.4f}--{s['k_only_max']:.4f} | {s['weight_only_min']:.4f}--{s['weight_only_max']:.4f} | "
            f"({s['largest_observed_k']}, {s['largest_observed_alpha']:g}) | {s['largest_observed_ndcg']:.4f} |")
    lines += ["", "The weight-sweep range includes single-branch ablations alpha=0 and 1.",
        "Largest observed means are selected on the same queries and are optimistic",
        "descriptions, not validated optimal settings or replacement test results.", ""]
    for s in summaries:
        selected = [r for r in rows if r["dataset"] == s["dataset"] and r["group"] == s["group"] and r["metric"] == "nDCG@10"]
        lines += [f"## {s['dataset']} / {s['group']}", "",
            "| k / lexical weight | 0 | .25 | .5 | .75 | 1 |", "|---|---:|---:|---:|---:|---:|"]
        for k in K_VALUES:
            vals = [next(r["mean"] for r in selected if r["k"] == k and r["alpha"] == a) for a in ALPHAS]
            lines.append(f"| {k} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
        lo, hi = s["largest_observed_pointwise_delta_ci"]
        slo, shi = s["largest_observed_simultaneous_delta_ci"]
        lines += ["", f"Largest observed minus default: {s['largest_observed_delta']:+.4f}; "
            f"pointwise 95% CI [{lo:+.4f}, {hi:+.4f}], within-grid simultaneous band [{slo:+.4f}, {shi:+.4f}].",
            f"Grid cells with a positive lower simultaneous bound: {s['grid_cells_with_positive_simultaneous_delta']}.", ""]
    lines += ["## Interpretation and limits", "",
        "The equal-weight k sweep changes full LitSearch and author-written means only",
        "modestly over this grid; manual TREC varies more. Weight changes are material,",
        "and the largest observed settings differ between the priority subsets:",
        "author-written LitSearch favors more lexical weight, manual TREC more dense",
        "weight. This does not imply that either preference generalizes to new queries.",
        "No alternative has a positive lower simultaneous bound against the default",
        "within the corresponding group. That is not an equivalence or optimality test.",
        "We retain the original default to avoid replacing it with a test-selected model.",
        "Future optimization needs a separate development set and untouched test set.", "",
        "Public qrels remain incomplete. Unjudged documents receive zero gain, without",
        "being verified negatives; recall is against known qrels. TREC uses the existing",
        "judgment-pool-derived reduced corpus, not the official full-corpus task.",
        "Results are conditional on fixed candidate depth, dense similarity floor and",
        "plain dense rankings. They do not establish RAG answer quality or isolate",
        "the benefit of self-evaluation/external search.", "",
        "## Reproduce", "", "```powershell",
        "python benchmarks/public_retrieval/rrf_sensitivity.py", "```", "",
        "Inputs are hashed before analysis and checked unchanged afterward.",
        "Assertions reproduce every original default ranking, per-query metric and",
        "both single-branch endpoints. See manifest.json and input_sha256.json.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf8")


def package(out, source, hashes):
    """Copy analysis and exact public ID/qrels inputs, never vectors or corpus text."""
    target = ROOT / "Report/supplement/rrf_sensitivity_20261102"
    target.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file():
            shutil.copy2(p, target / p.name)
    for name in hashes:
        dest = target / "inputs" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, dest)
    for relative in ("benchmarks/public_retrieval/rrf_sensitivity.py", "benchmarks/public_retrieval/run.py",
                     "tests/test_rrf_sensitivity.py"):
        dest = target / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    (target / "README.md").write_text(
        "# RRF sensitivity supplement\n\n"
        "See REPORT.md for the protocol, complete grid and limits.\n"
        "inputs/ contains the original released query qrels and frozen public ranking\n"
        "checkpoints used here, with run manifests and reference metric rows. No PDFs,\n"
        "corpus passages, vector stores, credentials or new LLM labels are included.\n\n"
        "Reproduce from the project root without APIs or model loading:\n\n"
        "```powershell\n"
        "python benchmarks/public_retrieval/rrf_sensitivity.py --source Report/supplement/rrf_sensitivity_20261102/inputs --out output/benchmarks/rrf_sensitivity_recomputed --no-package\n"
        "```\n\n"
        "The normal run also refreshes this package. Use --no-package for an independent\n"
        "recomputation that leaves this snapshot unchanged. source/ contains exact code\n"
        "copies; input_sha256.json hashes every consumed source input. SHA256.json hashes\n"
        "the package files. All conclusions concern plain RRF and fixed candidate lists.\n",
        encoding="utf8")
    # Hash all files other than the hash listing itself.
    write_json(target / "SHA256.json", {p.relative_to(target).as_posix(): digest(p)
        for p in sorted(target.rglob("*")) if p.is_file() and p.name != "SHA256.json"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--no-package", action="store_true", help="Do not refresh the manuscript supplement")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if not out.is_relative_to(ROOT / "output/benchmarks") or out == source or source.is_relative_to(out):
        raise ValueError("Use a separate analysis directory inside output/benchmarks")
    out.mkdir(parents=True, exist_ok=True)
    hashes, rows, summaries, per_query = {}, [], [], []
    for dataset in ("litsearch", "trec_rag_2024"):
        queries, values, pq, rankings, diagnostics, inputs = load_dataset(source, dataset)
        per_query.extend(pq)
        hashes.update(inputs)
        groups = {"all": list(range(len(queries)))}
        if dataset == "litsearch":
            groups["author_written"] = [i for i, q in enumerate(queries) if q["group"].startswith("manual_")]
        else:
            for group in ("fully_manual_27", "umbrela_extra_23"):
                groups[group] = [i for i, q in enumerate(queries) if q["group"] == group]
        write_json(out / f"{dataset}_top10.json", rankings)
        write_csv(out / f"{dataset}_candidate_diagnostics.csv", diagnostics)
        for group, selected in groups.items():
            r, summary = group_results(dataset, group, queries, values, selected)
            rows.extend(r)
            summaries.append(summary)
            print(dataset, group, json.dumps(summary), flush=True)
    # Verify input immutability after analysis, including qrels and every ranking.
    assert all(digest(source / name) == h for name, h in hashes.items())
    write_csv(out / "metrics_summary.csv", rows)
    write_csv(out / "metrics_per_query.csv", per_query)
    write_json(out / "summary.json", summaries)
    write_json(out / "input_sha256.json", hashes)
    write_json(out / "manifest.json", {"seed": SEED, "bootstrap_repetitions": BOOTSTRAPS,
        "grid_revision_date": "2026-11-03",
        "grid_revision_context": "Author-requested revised smoothing grid after previous evaluation and sensitivity outputs were seen; fixed before rerun; retrospective analysis, no held-out optimization.",
        "source": str(source.relative_to(ROOT)), "k_values": K_VALUES, "alphas": ALPHAS,
        "default": {"k": 60, "alpha": .5}, "branch": "plain dense, no reranker",
        "candidate_depth_per_branch": 100, "cutoff": 10, "metrics": METRICS,
        "tie_break": "ascending document ID", "missing_branch_contribution": 0,
        "qrels": "unchanged released qrels; unjudged receives zero gain, not a verified negative",
        "uncertainty": "10000 paired query-bootstrap samples; pointwise percentile 95% CIs; nDCG grid contrasts additionally have centered max-absolute bootstrap simultaneous bands within each dataset/group, not across groups",
        "limitations": "Post-hoc analysis of already evaluated queries; no held-out optimization. Query independence is approximate; target papers may be shared. TREC uses a judgment-pool-derived reduced corpus. This grid analyzes plain RRF only, conditional on fixed top-100 candidate rankings and similarity floor.",
        "network_calls": 0, "llm_calls": 0, "new_queries_or_labels": 0,
        "baseline_rankings_reproduced": True, "baseline_per_query_metrics_reproduced": True,
        "endpoint_rankings_reproduced": True, "source_inputs_unchanged": True,
        "source_code_sha256": digest(Path(__file__)),
        "scoring_source_sha256": digest(Path(metrics.__code__.co_filename)),
        "python_version": sys.version, "numpy_version": np.__version__})
    figures(out, rows)
    reports(out, rows, summaries)
    if not args.no_package:
        package(out, source, hashes)


if __name__ == "__main__":
    # Permit direct execution without installing the repository as a package.
    main()
