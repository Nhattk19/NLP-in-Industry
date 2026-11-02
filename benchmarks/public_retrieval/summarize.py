"""Validate completed runs and write a Vietnamese overview without new retrieval."""
import csv
import json
from pathlib import Path
from run import BASE, METHODS, read, save


def table(dataset,group):
    with (BASE/dataset/'metrics_summary.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    lines=['| Phương pháp | MRR@10 | Recall@10 | Hit@10 | nDCG@10 |',
           '|---|---:|---:|---:|---:|']
    for method in METHODS:
        values=[float(next(r['mean'] for r in rows if r['group']==group and r['method']==method and r['metric']==metric))
                for metric in ['MRR@10','Recall@10','Hit@10','nDCG@10']]
        lines.append('| '+method+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    return '\n'.join(lines)


def validate(dataset):
    out=BASE/dataset
    assert (out/'completed.json').exists()
    corpus=read(out/'corpus.json');ids={r['id'] for r in corpus}
    assert len(ids)==len(corpus)
    queries=read(out/'queries.json');empty=[];missing_gold={}
    for q in queries:
        result=read(out/'runs'/(q['id']+'.json'))
        assert set(result['rankings'])==set(METHODS)
        for ranking in result['rankings'].values():
            assert len(ranking)==len(set(ranking))
            assert set(ranking)<=ids
        if not result['rankings']['dense']:empty.append(q['id'])
        missing=set(q['qrels'])-ids
        assert all(q['qrels'][d]==0 for d in missing)
        if missing:missing_gold[q['id']]=sorted(missing)
    save(out/'validation.json',{'corpus':len(ids),'queries':len(queries),'all_run_ids_valid':True,
         'no_duplicate_results':True,'all_positive_gold_ids_in_corpus':True,
         'unavailable_zero_grade_ids_by_query':missing_gold,
         'empty_dense_after_fixed_similarity_floor':empty,
         'note':'Empty dense lists retained without duplicating the lexical branch; no queries dropped.'})
    return queries


def main():
    for dataset in ['litsearch','trec_rag_2024']:validate(dataset)
    trec=read(BASE/'trec_rag_2024/prepared.json')
    parts=['# Kết quả benchmark retrieval công khai', '',
       'Đã chạy bốn cấu hình: BM25, dense, Hybrid-RRF và Hybrid-RRF có reranker.',
       'Không gọi LLM để sinh câu hỏi, gán nhãn hay sinh câu trả lời trong lần chạy này.', '',
       '## LitSearch: full corpus và full queries', '',
       '- 64.183 paper; 597 query; biểu diễn title + abstract.',
       '- Gold là target-paper IDs gốc, không phải tập relevance đầy đủ trên mọi paper.',
       '- 351 query inline-citation và 246 query author-written; không gọi toàn bộ tập là human-written.', '',
       table('litsearch','all'), '',
       '### Riêng 246 query do tác giả viết', '',table('litsearch','author_written'), '',
       '## TREC RAG 2024: reduced-corpus experiment', '',
       f"- {trec['corpus']:,} passage; 50 topic: 27 fully manual và 23 topic UMBRELA chọn bằng seed 20261029.",
       f"- {trec['found_assessed_segments']:,} passage có nhãn (gồm nhãn 0) và {trec['random_distractors']:,} passage lấy mẫu ngẫu nhiên.",
       '- Quét đủ 113.520.750 segment trong 60 shard; nguồn mirror và revision lưu trong manifest.',
       '- Không phải đánh giá official trên full corpus TREC; corpus được chọn từ judgment pools rồi bổ sung distractor.', '',
       '### Kết quả chính: 27 topic fully manual', '',table('trec_rag_2024','fully_manual_27'), '',
       '### Toàn bộ 50 topic (trộn hai nguồn nhãn)', '',table('trec_rag_2024','all'), '',
       '### Riêng 23 topic UMBRELA', '',table('trec_rag_2024','umbrela_extra_23'), '',
       '## Phân tích kết quả và cách dùng trong revision', '',
       '1. Trên toàn bộ LitSearch, RRF cải thiện nDCG@10 từ 0,3771/0,3508 '
       '(BM25/dense) lên 0,4407. Paired 95% query-bootstrap CIs của chênh lệch '
       'với BM25 và dense lần lượt là [0,0417; 0,0856] và [0,0647; 0,1144]. '
       'RRF có reranker đạt 0,4558. Các khoảng này là thăm dò, chưa điều chỉnh multiple comparisons.', '',
       '2. Trong 246 query author-written, RRF đạt 0,5185 so với BM25 0,4827 và dense 0,3647. '
       'Chênh lệch RRF so với BM25 có CI [-0,0002; 0,0711], nên chưa đủ khẳng định '
       'vượt BM25 trên subset này. RRF có reranker đạt trung bình 0,5439.', '',
       '3. Với 27 topic TREC fully manual, RRF đạt nDCG@10 0,3691 so với BM25 0,2713 '
       'và dense 0,3426. CI chênh lệch RRF - BM25 là [0,0436; 0,1540], còn '
       'RRF - dense là [-0,0314; 0,0823]. Do đó, chỉ có bằng chứng rõ trong phép '
       'so sánh với BM25 theo CI này; mức tăng so với dense còn chưa chắc chắn. '
       'Reranker nâng trung bình lên 0,4241, với CI chênh lệch so với plain RRF '
       '[0,0278; 0,0822].', '',
       '4. Trên cả 50 topic TREC, dense 0,3326 và RRF 0,3312 gần như ngang nhau. '
       'Trong riêng 23 topic UMBRELA, dense đạt 0,3209, cao hơn RRF 0,2866 và '
       'RRF có reranker 0,3015 về trung bình. Không được viết hybrid luôn thắng dense '
       'hoặc bỏ subset này vì kết quả không thuận lợi.', '',
       '5. Reranking có chi phí đáng kể: median retrieval LitSearch tăng từ 0,391 s '
       '(plain RRF) lên 5,713 s; TREC tăng từ 0,208 s lên 8,692 s. Đây là '
       'retrieval-only component-summed timings, không phải thời gian trả lời chatbot.', '',
       '6. Recall@10 của TREC thấp không đồng nghĩa Hit@10 thấp: mỗi topic có nhiều '
       'passage được gán nhãn relevant nhưng top-10 chỉ chứa tối đa 10 passage. '
       'Cần đọc thêm Recall@100 trong CSV và không so trực tiếp giá trị Recall '
       'giữa LitSearch và TREC như thể hai bài toán tương đương.', '',
       'Câu kết luận gợi ý: Hybrid-RRF improves aggregate paper retrieval on LitSearch '
       'and outperforms BM25 in the reduced-corpus TREC experiment, while its advantage '
       'over dense passage retrieval is not consistent across topic/label subsets. '
       'Dense-branch reranking improves mean ranking quality at a substantial latency cost.', '',
       '## Cấu hình và cách đọc kết quả', '',
       '- MiniLM all-MiniLM-L6-v2, CUDA, tối đa 256 token; cosine Chroma HNSW.',
       '- Mỗi nhánh lấy tối đa 100 candidate; dense cosine floor 0,1; RRF k=60.',
       '- FlashRank ms-marco-MiniLM-L-12-v2 rerank nhánh dense trước RRF, không phải sau fusion.',
       '- Reranker CPU ONNX: 4 intra-op threads, 1 inter-op thread, microbatch 8, max length 512.',
       '- Bảng trên: relevant khi grade >=1; nDCG dùng gain 2^grade-1.',
       '- CSV có thêm cutoff 5/20/100, linear-gain nDCG và sensitivity grade >=2 cho TREC.',
       '- Recall tính trên known qrels, không phải toàn bộ relevant items ngoài tập đã gán nhãn.',
       '- Unjudged nhận gain 0 khi tính metric; điều này không khẳng định chúng thật sự irrelevant.',
       '- Khoảng tin cậy và paired differences: 10.000 query bootstrap, seed cố định, chưa điều chỉnh multiple comparisons; query có thể phụ thuộc nhau.', '',
       '## Ngoại lệ dữ liệu đã được người dùng duyệt', '',
       '`msmarco_v2.1_doc_01_23#45_67` có trong qrels nhưng không tìm thấy sau khi quét toàn corpus. '
       'Mọi grade của ID này trong 50 topic đã chọn đều là 0. Theo xác nhận của người dùng, bỏ yêu cầu đưa ID đó vào corpus, '
       'giữ nguyên qrels gốc và thêm distractor để đủ 64.200 passage. Không mất passage có nhãn relevant nào.', '',
       '## Thời gian và phạm vi', '',
       'Latency CSV là warm retrieval component timings; hybrid được cộng từ các thành phần tuần tự. '
       'Không gồm download, model loading, xây index, LLM hoặc external search. '
       'Đầu lần chạy LitSearch có tải/giải nén TREC đồng thời; thời gian quan sát có thể bị ảnh hưởng bởi tải nền. '
       'Không diễn giải đây là end-to-end latency của chatbot.', '',
       'Hai benchmark cung cấp bằng chứng bổ sung: paper retrieval trong miền ML/NLP và passage retrieval trên web corpus rút gọn. '
       'Chúng chưa thay thế đánh giá trực tiếp full-text passage retrieval trên corpus NLP của đồ án, '
       'và không tự chứng minh chất lượng generation hoặc lợi ích của vòng agent.', '',
       '## File kết quả', '',
       '- `litsearch/RESULTS.md`, `trec_rag_2024/RESULTS.md`: bảng theo nhóm.',
       '- `metrics_summary.csv`, `paired_differences.csv`, `metrics_per_query.csv`: metric và CI.',
       '- `latency_summary.csv`: median/p95/mean theo thành phần.',
       '- `runs/`: ranking từng query; `trec_format/`: run/qrels để kiểm tra độc lập.',
       '- `run_manifest.json`, `prepared.json`, `validation.json`: cấu hình, corpus, kiểm tra.',
       '- `retrieval_comparison.png`: biểu đồ; với TREC biểu đồ tổng trộn hai nguồn nhãn.',
       '- Script chạy và protocol: `benchmarks/public_retrieval/README.md`.', '']
    (BASE/'SUMMARY_VI.md').write_text('\n'.join(parts),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,dataset,group,title in zip(axes,['litsearch','trec_rag_2024'],
            ['author_written','fully_manual_27'],['LitSearch: 246 author-written queries','TREC reduced corpus: 27 fully manual topics']):
        with (BASE/dataset/'metrics_summary.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        selected=[next(r for r in rows if r['group']==group and r['method']==m and r['metric']=='nDCG@10') for m in METHODS]
        means=np.array([float(r['mean']) for r in selected])
        ax.bar(range(4),means,color=['#8294a4','#536d82','#247a72','#d18b37'])
        ax.errorbar(range(4),means,yerr=[means-np.array([float(r['ci_low']) for r in selected]),
             np.array([float(r['ci_high']) for r in selected])-means],fmt='none',capsize=4,ecolor='#222222')
        ax.set_xticks(range(4),['BM25','Dense','RRF','RRF + rerank'],rotation=15)
        ax.set_ylim(0,1);ax.set_ylabel('nDCG@10 (exponential gain)');ax.set_title(title,fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Human-origin query / judgment subsets — exploratory 95% CIs',fontsize=12)
    fig.tight_layout();fig.savefig(BASE/'key_subsets.png',dpi=180);fig.savefig(BASE/'key_subsets.svg');plt.close(fig)
    print('VALIDATED BOTH BENCHMARKS; SUMMARY_VI.md written',flush=True)


if __name__=='__main__':main()
