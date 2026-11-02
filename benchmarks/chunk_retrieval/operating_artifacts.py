"""Audit completed online measurements and create paper assets, without API calls."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT / "output/benchmarks/online_qa_cost_20261027"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    plan=read(OUT / "queries.json")
    responses=[read(f) for f in sorted((OUT / "responses").glob("qa*.json"))]
    assert len(responses)==30
    assert {r["case"]["query_id"] for r in responses}=={q["query_id"] for q in plan}
    call_count=attempts=0
    for r in responses:
        assert r["output"]["success"]
        assert r["initial_chunks"]==204539
        assert sum(c["elapsed_ms"] for c in r["calls"])<=r["end_to_end_ms"]+1
        for c in r["calls"]:
            assert c["status"]=="ok"
            assert c["input_tokens"]+c["output_tokens"]==c["total_tokens"]
            assert c["reasoning_tokens"]<=c["output_tokens"]
            expected=((c["input_tokens"]-c["cached_tokens"])*.30+c["cached_tokens"]*.03+c["output_tokens"]*2.50)/1e6
            assert abs(expected-c["estimated_usd"])<1e-12
            assert c["http_attempts"]
            attempts+=len(c["http_attempts"]);call_count+=1
    assert len(list((OUT / "calls").glob("qa*.json")))==call_count
    verification=read(OUT / "source_verification.json")
    assert verification["unchanged"] and verification["before"]==verification["after"]
    assert (OUT / "source_verification.json").stat().st_mtime>=max(
        f.stat().st_mtime for f in (OUT / "responses").glob("qa*.json"))
    # Artifact-only credential check. Never inspect or copy the key file.
    for f in (OUT / "calls").glob("*.json"):
        assert not re.search(r"AIza[0-9A-Za-z_-]{30,}",f.read_text(encoding="utf-8"))
    validation={"formal_queries":30,"logical_calls":call_count,"http_attempts":attempts,
        "usage_arithmetic_valid":True,"reasoning_not_double_counted":True,
        "listed_price_calculation_valid":True,"source_document_metadata_unchanged":True,
        "source_embeddings_not_hashed":True,"credential_pattern_found_in_call_artifacts":False}
    (OUT / "VALIDATION.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    manifest=read(OUT / "manifest.json")
    manifest["source_hashes"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "src/agent/agent.py",ROOT / "src/web/core/rag.py",
            *sorted((ROOT / "src/agent/nodes").glob("*.py")),
            ROOT / "benchmarks/chunk_retrieval/operating_cost.py"]}
    (OUT / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    formatting=[]
    expected_keys={"intent_classification":"intent","answer_evaluation":"score",
        "conversation_rewrite":"standalone_question","arxiv_query_rewrite":"search_query"}
    for r in responses:
        for c in r["calls"]:
            if c["stage"] not in expected_keys:continue
            text=str(c["response"]).strip()
            if "```json" in text:text=text.split("```json",1)[1].split("```",1)[0].strip()
            elif "```" in text:text=text.split("```",1)[1].split("```",1)[0].strip()
            try:
                parsed=json.loads(text); valid=expected_keys[c["stage"]] in parsed
            except (ValueError,TypeError):valid=False
            formatting.append({"query_id":c["query_id"],"call_index":c["call_index"],
                "stage":c["stage"],"valid_json_with_expected_key":valid,
                "finish_reason":c["response_metadata"].get("finish_reason")})
    with (OUT / "json_format_diagnostics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(formatting[0]));w.writeheader();w.writerows(formatting)

    groups=table(OUT / "summary.csv"); all_result=next(r for r in groups if r["group"]=="all")
    stage_rows=table(OUT / "llm_stage_summary.csv")
    names={"all":"All","no_external":"No external search","external_attempted":"External attempted"}
    tex=[r"\begin{tabular}{@{}lrrrr@{}}",r"\toprule",
        r"Requests & $n$ & E2E med./p95 (s) & Calls & Est. USD \\",r"\midrule"]
    for g in groups:
        if g["group"] in names:
            tex.append(f"{names[g['group']]} & {g['queries']} & {float(g['median_e2e_seconds']):.2f} / "
                f"{float(g['p95_e2e_seconds']):.2f} & {float(g['mean_logical_calls']):.2f} & "
                f"{float(g['mean_estimated_usd']):.5f} " + r"\\")
    tex += [r"\bottomrule",r"\end{tabular}"]
    (OUT / "online_cost_table.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    raw=table(OUT / "per_query.csv")
    for flag,label,color in [("False","No external search","#2878b5"),("True","External attempted","#d87523")]:
        subset=[r for r in raw if r["external_attempted"]==flag]
        if subset:
            axes[0].scatter([int(r["query_id"][2:]) for r in subset],
                [float(r["end_to_end_ms"])/1000 for r in subset],label=label,color=color,s=30)
    axes[0].set_yscale("log");axes[0].set_xlabel("Prespecified query order")
    axes[0].set_ylabel("Server-side E2E seconds (log scale)");axes[0].legend(fontsize=8)
    stage_names={"intent_classification":"Intent","answer_generation":"Answer",
        "answer_evaluation":"Evaluation","conversation_rewrite":"Conversation rewrite",
        "arxiv_query_rewrite":"arXiv rewrite","answer_continuation":"Continuation"}
    labels=[stage_names.get(r["stage"],r["stage"]) for r in stage_rows]
    med=[float(r["median_ms"])/1000 for r in stage_rows]
    upper=[(float(r["p95_ms"])-float(r["median_ms"]))/1000 for r in stage_rows]
    axes[1].barh(labels,med,color="#2878b5",xerr=[[0]*len(med),upper],capsize=3)
    axes[1].set_xlabel("LLM call seconds: median with p95 cap")
    for ax in axes:ax.grid(alpha=.2,axis="both");ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(OUT / "online_cost.png",dpi=200)
    fig.savefig(OUT / "online_cost.svg");plt.close(fig)

    diagnostic=read(OUT / "diagnostic_pilot_reference_cycle/EXCLUSION.json")
    connectivity=read(OUT / "connectivity_diagnostics.json")
    formal_cost=sum(float(r["estimated_usd"]) for r in raw)
    total=formal_cost+diagnostic["listed_price_estimated_usd"]+connectivity["listed_price_estimated_usd"]
    bad_format=sum(not r["valid_json_with_expected_key"] for r in formatting)
    lines=["# Phân tích latency và chi phí QA online", "",
        "Đã hoàn tất 30 câu cố định: 18 câu độc lập, 6 follow-up và 6 câu thăm dò coverage gần đây. "
        "Đo qua web wrapper thật với Gemini 2.5 Flash, không ép tìm ngoài, không thay prompt hoặc chính sách evaluator.","",
        "| Nhóm | Số câu | E2E median / p95 (giây) | Calls trung bình | Input / output token trung bình | USD ước tính/câu |",
        "|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g['group']} | {g['queries']} | {float(g['median_e2e_seconds']):.2f} / {float(g['p95_e2e_seconds']):.2f} | "
            f"{float(g['mean_logical_calls']):.2f} | {float(g['mean_input_tokens']):.1f} / "
            f"{float(g['mean_output_tokens_including_reasoning']):.1f} | {float(g['mean_estimated_usd']):.6f} |")
    lines += ["", "## Diễn giải", "",
        f"- Đã ghi {call_count} logical call và {attempts} HTTP attempt; không có call formal lỗi API hoặc thiếu usage.",
        "- Output token đã bao gồm reasoning trong SDK hiện dùng. Có lưu riêng reasoning và phần output nhìn thấy; không cộng reasoning lần thứ hai.",
        "- E2E là thời gian xử lý phía server từ trước rewrite tới sau khi chuẩn hóa kết quả. Không gồm khởi tạo/model warmup, reset corpus thử nghiệm, giao diện/browser hoặc thời gian tới token đầu tiên.",
        "- Dùng private index, xóa chunk mới trước mỗi câu và reset BM25 ngoài đồng hồ khi cần. Logical corpus được reset, nhưng lịch sử xóa/tombstone của HNSW có thể còn.",
        "- Các stage lồng nhau, ví dụ dense chứa rerank; không cộng mọi dòng trong local_stage_summary.csv để tính E2E.",
        "- External_attempted lấy từ trace thực tế. Có ingest không đồng nghĩa bằng chứng mới được trích dẫn; per_query.csv có external_chunks_cited để phân biệt.",
        f"- Kiểm tra định dạng JSON hậu nghiệm phát hiện {bad_format} output không parse được theo kiểm tra strict/khóa mong đợi. "
        "Đây là chẩn đoán định dạng, không phải lỗi HTTP; các call vẫn được tính token/chi phí. Không dùng nó thay đánh giá chất lượng câu trả lời.",
        "- Tập câu hỏi synthetic có chủ ý, không phải traffic đại diện; follow-up phụ thuộc câu cha. Một quan sát/câu, p95 với mẫu nhỏ chỉ mang tính mô tả.",
        "", "## Chi phí và phần loại khỏi benchmark", "",
        f"- Formal 30 câu: **{formal_cost:.6f} USD** theo bảng giá niêm yết standard, không phải hóa đơn thực trả.",
        f"- Pilot kỹ thuật bị loại: {diagnostic['recorded_calls']} call đã ghi, khoảng {diagnostic['listed_price_estimated_usd']:.6f} USD. "
        "Lý do: wrapper ban đầu có thể giữ index BM25 cũ; đã sửa trước pilot/formal hợp lệ. Dữ liệu chẩn đoán giữ trong diagnostic_pilot_reference_cycle/.",
        f"- Hai call kiểm tra kết nối: khoảng {connectivity['listed_price_estimated_usd']:.6f} USD, tách riêng.",
        f"- Tổng usage có ghi nhận của cả đợt: khoảng **{total:.6f} USD**. Phần request bị ngắt không có response không thể suy ra hóa đơn.",
        "- Đơn giá 27/10/2026: input 0,30 USD/M, output gồm reasoning 2,50 USD/M, cached input 0,03 USD/M. Không gồm phần cứng/điện, free quota/credit hoặc thuế.",
        "", "## Artifact", "",
        "- REPORT.md: protocol và bảng số đo đầy đủ.",
        "- per_query.csv, per_call.csv, llm_stage_summary.csv, local_stage_summary.csv: số liệu.",
        "- calls/, responses/: prompt, response, usage và trace dùng kiểm tra lại, không chứa API credential.",
        "- online_cost_table.tex, online_cost.png/svg: bảng và hình để đưa vào bài.",
        "- VALIDATION.json, source_verification.json, manifest.json: kiểm tra và cấu hình.", ""]
    (OUT / "PHAN_TICH_KET_QUA.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"validated_queries":30,"formal_cost_usd":formal_cost,
        "all_recorded_experiment_cost_usd":total,"logical_calls":call_count,"http_attempts":attempts}))


if __name__=="__main__":main()
