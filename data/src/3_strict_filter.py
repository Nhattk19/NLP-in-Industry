import os
import json
import re
from tqdm import tqdm
import config

INPUT_FILE = os.path.join(config.DATA_RAW_DIR, "2_map.jsonl")
OUTPUT_FILE = os.path.join(config.DATA_RAW_DIR, "3_final_nlp_papers.jsonl")

def calculate_nlp_score(doc):
    title = str(doc.get("title", "")).lower()
    abstract = str(doc.get("abstract", "")).lower()
    venue = str(doc.get("venue", "")).lower()
    
    score = 0
    
    # 1. Chấm điểm Venue
    for pattern in config.PURE_NLP_VENUES:
        if re.search(pattern, venue):
            score += 20
            break
    for pattern in config.GENERAL_AI_VENUES:
        if re.search(pattern, venue):
            score += 5
            break
            
    # 2. Chấm điểm Title (Trọng số cao: Strong = +5, Context = +2)
    for pattern in config.STRONG_KEYWORDS:
        if re.search(pattern, title): score += 5
    for pattern in config.CONTEXT_KEYWORDS:
        if re.search(pattern, title): score += 2

    # 3. Chấm điểm Abstract (Trọng số thấp hơn: Strong = +2, Context = +1)
    for pattern in config.STRONG_KEYWORDS:
        if re.search(pattern, abstract): score += 2
    for pattern in config.CONTEXT_KEYWORDS:
        if re.search(pattern, abstract): score += 1

    # 4. Chấm điểm Citations & References mạng lưới
    # CHỈ DÙNG STRONG_KEYWORDS để tránh nhiễu từ các từ chung chung
    network_matches = 0
    
    network = doc.get("network", {})
    citations = network.get("citations", [])
    if isinstance(citations, list):
        for cit in citations:
            cit_title = str(cit.get("title", "")).lower()
            if any(re.search(p, cit_title) for p in config.STRONG_KEYWORDS):
                network_matches += 1

    references = network.get("references", [])
    if isinstance(references, list):
        for ref in references:
            ref_title = str(ref.get("title", "")).lower()
            if any(re.search(p, ref_title) for p in config.STRONG_KEYWORDS):
                network_matches += 1

    # Thưởng tối đa 15 điểm cho mạng lưới trích dẫn (1 điểm/bài)
    score += min(network_matches, 15)
            
    # 5. Penalties (Kiểm tra chéo title và abstract)
    text_content = f"{title} {abstract}"
    if "image" in text_content and "language" not in text_content and "text" not in text_content:
        score -= 10
    if "clinical" in text_content and "nlp" not in text_content:
        score -= 20
    if "network" in text_content and "neural" not in text_content and "language" not in text_content:
        score -= 5
        
    return score

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Không tìm thấy {INPUT_FILE}.")
        return

    # Tăng Threshold lên 15 để đảm bảo chất lượng
    THRESHOLD = 15 
    total_in = 0
    total_out = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Scoring Papers"):
            total_in += 1
            try:
                doc = json.loads(line)
                score = calculate_nlp_score(doc)
                
                if score >= THRESHOLD:
                    doc['nlp_score'] = score
                    f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    total_out += 1
            except: continue
                
    print(f"\n📊 THỐNG KÊ (Ngưỡng {THRESHOLD} điểm):")
    print(f"- Input (Đã map data): {total_in}")
    print(f"- Output (NLP Valid):  {total_out}")
    if total_in > 0:
        print(f"- Tỷ lệ giữ lại:       {total_out/total_in*100:.2f}%")

if __name__ == "__main__":
    main()