import json
import torch
from transformers import pipeline, AutoTokenizer

# ============================================================
# 1. Load model (PyTorch)
# ============================================================

model_name = "TimSchopf/nlp_survey_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline(
    "text-classification",
    model=model_name,
    tokenizer=tokenizer,
    framework="pt",      # Force PyTorch
    device=0 if torch.cuda.is_available() else -1,  # Sử dụng GPU nếu có
    truncation=True
)

print("Model loaded successfully.")

# ============================================================
# 2. Load JSONL file
# ============================================================
file_path = "data/data_raw/3_final_nlp_papers.jsonl"
papers = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        papers.append(json.loads(line))

print(f"Loaded {len(papers)} papers.")

# ============================================================
# 3. Prepare input text & Predict (batch processing)
# ============================================================

inputs = []
for paper in papers:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    
    # Ghép chuỗi đầu vào cho model
    text = title + tokenizer.sep_token + abstract
    inputs.append(text)

batch_size = 16  # Kích thước batch an toàn
pred_labels = []

print("Running predictions... This might take a while.")

for i in range(0, len(inputs), batch_size):
    batch = inputs[i:i + batch_size]
    predictions = classifier(batch)
    
    for pred in predictions:
        # Chuyển đổi nhãn của model thành boolean (True/False)
        is_survey = True if pred["label"].lower() == "survey" else False
        pred_labels.append(is_survey)

print("Prediction completed.")

# ============================================================
# 4. Save result to a new JSONL file
# ============================================================

output_path = "data/data_raw/05_final_check_survey_data.jsonl"

with open(output_path, "w", encoding="utf-8") as out_f:
    for idx, paper in enumerate(papers):
        # Ghi đè kết quả dự đoán trực tiếp lên biến "is_survey" cũ
        paper["is_survey"] = pred_labels[idx]
        
        # Ghi dictionary thành một dòng JSON hợp lệ vào file mới
        out_f.write(json.dumps(paper, ensure_ascii=False) + "\n")

print(f"Successfully saved all updated data to {output_path}")