# NLP Paper Classification Evaluation

Hệ thống đánh giá tự động phân loại papers là NLP hay không, sử dụng 2 mô hình AI khác nhau (Gemini & Groq), so sánh kết quả để tìm các papers cần review thủ công.

---

## 1. Setup & Configuration

### 1.1 Google Gemini - Vertex AI Setup

#### Bước 1: Tạo Google Cloud Project
1. Truy cập [Google Cloud Console](https://console.cloud.google.com/)
2. Tạo project mới hoặc chọn project hiện tại
3. Enable API: **Vertex AI API** và **Cloud Compute API**

#### Bước 2: Tạo Service Account & Credentials
1. Vào **IAM & Admin** → **Service Accounts**
2. Click **Create Service Account**
3. Đặt tên (ví dụ: `nlp-eval`)
4. Grant roles: **Vertex AI User** hoặc **Vertex AI Service Agent**
5. Tạo JSON key:
   - Chọn service account vừa tạo
   - Vào tab **Keys**
   - Click **Add Key** → **Create new key** → JSON
   - File JSON sẽ tự download

#### Bước 3: Đặt Credentials File
- Copy file JSON vào thư mục eval:
  ```
  eval/
  ├── credential.json
  └── ...
  ```

**Lưu ý:** Đảm bảo `credential.json` có private key và project ID (dòng `"project_id"`)

### 1.2 Groq API Setup

#### Bước 1: Đăng ký Groq Account
1. Truy cập [Groq Console](https://console.groq.com/)
2. Tạo tài khoản mới hoặc đăng nhập
3. Vào **API Keys** section
4. Click **Create API Key**
5. Copy key (ví dụ: `gsk_abc123...`)

### 1.3 Environment Variables (.env)

Tạo file `.env` trong thư mục `eval/`:

```env
# Google Cloud / Vertex AI Credentials
GOOGLE_APPLICATION_CREDENTIALS=đường dẫn tới credential.json
GOOGLE_CLOUD_PROJECT=project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Groq API Key
GROQ_API_KEY=gsk_your_actual_key_here
```

**Giải thích:**
- `GOOGLE_APPLICATION_CREDENTIALS`: Đường dẫn tuyệt đối tới file `credential.json` (Service Account key từ GCP)
- `GOOGLE_CLOUD_PROJECT`: Project ID từ GCP Console (ví dụ: `flash-landing-492201-f4`)
- `GOOGLE_CLOUD_LOCATION`: Region cho Vertex AI (mặc định: `us-central1`)
- `GROQ_API_KEY`: API key từ Groq Console
- `.env` không nên commit vào Git (thêm vào `.gitignore`)

### 1.4 Installation

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Các thư viện chính:
# - google-cloud-aiplatform (Vertex AI)
# - groq (Groq API client)
# - pandas (xử lý CSV)
# - python-dotenv (load .env)
```

---

## 2. Cơ Chế Evaluation

### 2.1 Quy Trình Chạy

```
INPUT: final_cleaned_data.jsonl (cùng format từ data processing)
  ↓
[BATCH EVALUATION]
  • Chia papers thành batch (mỗi batch ~25 papers)
  • Batch 1 → Call Gemini → Parse kết quả → Save CSV
  • Batch 2 → Call Groq → Parse kết quả → Merge CSV
  • ...
  ↓
OUTPUT: eval_results.csv
```

### 2.2 Cách Gọi 2 Models

File `1_eval.py` dùng **Evaluator pattern** cho 2 LLM:

#### GeminiEvaluator (Vertex AI)
```python
evaluator = GeminiEvaluator()
# Gọi API Gemini thông qua Vertex AI
response = evaluator.generate(system_prompt, user_prompt)
```

**Đặc điểm:**
- Sử dụng Vertex AI GenerativeModel SDK
- Có retry logic với exponential backoff (khi quota exceeded)
- Model: `gemini-2.5-flash`

#### GroqEvaluator
```python
evaluator = GroqEvaluator(api_key=GROQ_KEY)
# Gọi API Groq qua chat completion interface
response = evaluator.generate(system_prompt, user_prompt)
```

**Đặc điểm:**
- Sử dụng Groq client library
- Model: `llama-3.1-8b-instant`
- Temperature = 0.0 (deterministic)

### 2.3 Prompt Design

Cả 2 models nhận **system prompt** chung (định nghĩa NLP, cách phân loại):

```
"A paper is NLP if it focuses on:
- Text classification, sentiment analysis
- Machine translation, language modeling
- Question answering, dialogue systems
- Information extraction, text generation
- Speech-to-text/text-to-speech (language-focused)

A paper is NOT NLP if it focuses on:
- Computer vision, pure signal processing
- General ML, robotics, systems, etc."
```

**Batch Processing:**
- Gộp nhiều papers thành 1 request (giảm API calls)
- Model trả về JSON array (thay vì từng object)
- Tiết kiệm chi phí ~75% so với single-paper mode

### 2.4 Output Format

**eval_results.csv:**
```
paper_id,groq_is_nlp,groq_confidence,gemini_is_nlp,gemini_confidence
265099508,False,0.95,True,0.8
5080441,True,0.95,True,1.0
...
```

**Các cột:**
- `paper_id`: ID paper từ input
- `groq_is_nlp`: Kết quả từ Groq (True/False)
- `groq_confidence`: Độ tự tin Groq (0.0 - 1.0)
- `gemini_is_nlp`: Kết quả từ Gemini (True/False)
- `gemini_confidence`: Độ tự tin Gemini (0.0 - 1.0)

---

## 3. Chạy Evaluation

### 3.1 Chỉ chạy Gemini

Mở `1_eval.py`, block Gemini sẽ active mặc định:
```python
# Option 1: Google Gemini / Vertex AI
if gemini_key:
    evaluator = GeminiEvaluator(gemini_key)
    evaluator_name = "gemini"
```

Block Groq sẽ comment lại.

Chạy:
```bash
python 1_eval.py
```

### 3.2 Chỉ chạy Groq

Mở `1_eval.py`:
1. Comment block Gemini (từ `if gemini_key:` đến `return`)
2. Uncomment block Groq (từ `elif groq_key:` đến `return`)

Chạy:
```bash
python 1_eval.py
```

### 3.3 Chạy cả 2 (tuần tự)

Chạy Gemini trước:
```bash
python 1_eval.py  # Gemini
```

Sau đó cập nhật code sang Groq (comment Gemini, uncomment Groq), chạy lại:
```bash
python 1_eval.py  # Groq
```

**Kết quả:** CSV sẽ merge cột `gemini_*` và `groq_*`

### 3.4 Resume từ Paper N

Tham số `start_from` trong `process_papers()`:
```python
process_papers(
    input_path=input_path,
    output_path=output_path,
    evaluator_name=evaluator_name,
    evaluator=evaluator,
    batch_size=25,
    start_from=0  # ← Bắt đầu từ paper index 0 (hoặc số khác nếu muốn resume)
)
```

---


## 4. So Sánh Kết Quả & Human Review

### 4.1 Tìm Papers Có Kết Quả Khác Nhau

Chạy script:
```bash
python 2_extract_differences.py
```

**Output:** `differences.csv` chứa papers mà Groq và Gemini không đồng ý:

```
paper_id,groq_is_nlp,gemini_is_nlp
265099508,False,True      ← Groq: NOT NLP, Gemini: NLP
280017901,False,True
284870624,False,True
...
```

**Ví dụ:** Nếu có 1,452 papers khác nhau từ 10,000 papers → cần review 14.5%

### 4.2 Phân Loại Disagreement

```python
# Type 1: Groq=False, Gemini=True
# → Gemini "aggressive" (hay classify là NLP)

# Type 2: Groq=True, Gemini=False  
# → Groq "aggressive"
```

### 4.3 Human Review Process

**Cách review:**
1. Mở `differences.csv`
2. Lấy `paper_id`, lọc từ original data
3. Kiểm tra title, abstract, venue
4. Quyết định: NLP hay NOT NLP
5. Ghi chú lý do (ví dụ: "borderline - tập trung vào application, không core NLP")

**Tạo ground truth:**
```python
# Sau khi human review, tạo file:
human_labels.csv
├── paper_id
├── human_is_nlp (True/False)
└── review_notes (optional)
```

### 4.4 Đánh Giá Performance

```python
# Sau khi có human labels:
- Accuracy Groq = TP+TN / Total
- Accuracy Gemini = TP+TN / Total
- Agreement rate = Số papers 2 model cùng đúng / Total
```

---

## 5. Troubleshooting

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-----------|---------|
| `credential.json` not found | Path trong .env sai | Dùng absolute path, kiểm tra tên file |
| `GOOGLE_APPLICATION_CREDENTIALS` invalid | JSON key không hợp lệ | Download lại từ GCP Console |
| `GOOGLE_CLOUD_PROJECT` not found | Quên thêm vào .env | Thêm `GOOGLE_CLOUD_PROJECT=your_project_id` vào .env |
| 429 Quota Exceeded | Quá nhiều API calls | Code có retry logic, mặc định wait 10s |
| `GROQ_API_KEY` not found | .env không load | Đảm bảo file `.env` cùng thư mục eval/ |
| Parse failed | Response format sai | Check response từ model (qua logs) |
| Memory exceed | Batch size quá lớn | Giảm batch_size từ 25 → 10 |

---

## 6. Files Structure

```
eval/
├── 1_eval.py                       # Main evaluation script
├── 2_extract_differences.py        # Script so sánh 2 kết quả models
├── eval_results.csv                # Output: kết quả từ 2 models (Gemini & Groq)
├── differences.csv                 # Output: papers có kết quả khác nhau
├── credential.json                 # Google Service Account key (KHÔNG commit)
├── .env                            # Environment variables (KHÔNG commit)
└── README.md                       # This file
```

---
