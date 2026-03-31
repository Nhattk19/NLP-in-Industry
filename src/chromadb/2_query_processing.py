## code này chạy trên kaggle được :v
# prompt engineering để LLM phân tích query, phân loại Broad/Specific/Explanatory..., rewrite lại query theo ngôn ngữ học thuật, và mở rộng thêm 5 keyword liên quan để tăng khả năng retrieve được nhiều paper hơn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import QUERY_PROCESSING_MODEL, DEVICE, MAX_NEW_TOKENS

# ================= CONFIG =================
# Bạn có thể dùng "Qwen/Qwen2.5-1.5B-Instruct" hoặc "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= INIT =================
print(f"Loading model: {QUERY_PROCESSING_MODEL} on {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(QUERY_PROCESSING_MODEL)
# Dùng bfloat16 để tiết kiệm bộ nhớ và tăng tốc độ trên RTX 3000 series
model = AutoModelForCausalLM.from_pretrained(
    QUERY_PROCESSING_MODEL, 
    torch_dtype="auto", 
    device_map="auto"
)

# ================= MAIN FUNCTION =================
def process_query(query):
    messages = [
        {"role": "system", "content": (
            "You are an Academic Search Expert. Your task is to rewrite queries "
            "into formal research language WITHOUT narrowing or changing the original scope."
        )},
        {"role": "user", "content": f"""Task: Transform the query into a formal academic search statement.
        
If topics, queries NOT related to NLP/AI (weather, politics, etc.), PLEASE return TYPE = "irrelevant" and there is no need for rewriting or expanding queries.
         
Example:
Input: "how does CNN work"
REWRITE: "Comprehensive structural analysis and operational principles of Convolutional Neural Networks"
EXPANSION: Feature maps, Pooling layers, Backpropagation, ReLu, Spatial hierarchy

Input: "{query}"
        
Instructions:
1. TYPE: Classify as broad, specific, explanatory, method, comparison, ambiguous, irrelevant, etc.
2. REWRITE: Maintain the EXACT scope of the original query but use formal academic terminology. Do not focus only on one sub-component.
3. EXPANSION: List 5 essential technical components of this architecture.

Format:
TYPE: [Type]
REWRITE: [Formal Query]
EXPANSION: [Keyword 1, Keyword 2, Keyword 3, Keyword 4, Keyword 5]"""}

    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=150, # Tăng một chút để REWRITE không bị cắt ngang
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    # Chỉ lấy phần text mới được sinh ra
    response = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_text = tokenizer.batch_decode(response, skip_special_tokens=True)[0]

    result = parse_output(output_text, query)
    # Logic xử lý Out-of-Scope
    if result["query_type"].lower() == "irrelevant":
        print(f"⚠️ Query '{query}' is out of NLP scope. Skipping processing.")
        return None # Hoặc trả về thông báo cho người dùng
    
    return result

# ================= PARSE =================
def parse_output(text, original_query):
    import re
    result = {
        "original_query": original_query,
        "query_type": "unknown",
        "rewrite": original_query,
        "expanded_query": original_query
    }
    
    # Regex xử lý linh hoạt hơn
    type_match = re.search(r"TYPE:\s*(.*)", text, re.IGNORECASE)
    rewrite_match = re.search(r"REWRITE:\s*(.*)", text, re.IGNORECASE)
    expansion_match = re.search(r"EXPANSION:\s*(.*)", text, re.IGNORECASE)

    # Tách nhỏ kết quả dựa trên dòng
    lines = text.strip().split('\n')
    for line in lines:
        if line.upper().startswith("TYPE:"):
            result["query_type"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("REWRITE:"):
            result["rewrite"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("EXPANSION:"):
            result["expanded_query"] = line.split(":", 1)[1].strip()

    return result

if __name__ == "__main__":
    query = "how transformer works"
    res = process_query(query)
    
    print("\n" + "="*30)
    for k, v in res.items():
        print(f"{k.upper()}: {v}")