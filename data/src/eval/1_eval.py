import os
import json
import csv
import time
import re
import random
from typing import Dict, Any
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel

system_prompt = """
You are an AI assistant specialized in academic Natural Language Processing (NLP) paper classification.

Your task:
Given a JSONL input where each line is a paper with metadata, classify whether the paper belongs to the field of Natural Language Processing (NLP).

Definition:
A paper is considered NLP if it focuses on processing, understanding, or generating human language (text or speech), including but not limited to:
- Text classification, sentiment analysis
- Machine translation
- Language modeling
- Question answering, dialogue systems
- Information extraction
- Text generation (NLG)
- Speech-to-text or text-to-speech (if language-focused)

A paper is NOT NLP if it mainly focuses on:
- Computer vision (images, video, object detection)
- Pure audio/signal processing without language understanding
- General machine learning without language-specific tasks
- Robotics, networking, systems, etc.

Instructions:
1. Read the following fields:
   - title
   - abstract
   - venue (if available)
2. Use semantic understanding (not keyword matching only).
3. Be robust to missing or noisy metadata.

Output format (STRICT JSON per line):
{
  "paper_id": "<paper_id>",
  "is_nlp": true/false,
  "confidence": float (0.0 to 1.0)
}

IMPORTANT for confidence:
- 0.8-1.0: Very confident this IS NLP (when is_nlp=true) or DEFINITIVELY NOT NLP (when is_nlp=false)
- 0.5-0.8: Somewhat uncertain
- 0.5: Borderline, could go either way

Constraints:
- Output ONLY valid JSON (no extra text).
- confidence MUST always be > 0.0, never 0.0
- Return confidence based on your certainty of the classification itself (true OR false)
- Do not hallucinate missing fields.

Now process the following input:
"""

class BaseEvaluator:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Phải implement hàm generate() cho từng model.")

class GeminiEvaluator(BaseEvaluator):
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        super().__init__(api_key, model_name)
        # Sử dụng Vertex AI 
        self.model = GenerativeModel(self.model_name)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Vertex AI GenerativeModel
        prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Retry logic với exponential backoff khi quota exceeded
        max_retries = 3
        base_wait_time = 10  # 10 giây
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_str = str(e)
                # Kiểm tra lỗi 429 (quota exceeded)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 2)  # Exponential backoff + jitter
                        print(f"  ⏳ Quota exceeded, waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ Quota limit - max retries exceeded")
                        raise
                else:
                    # Lỗi khác, không retry
                    raise

class GroqEvaluator(BaseEvaluator):
    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        super().__init__(api_key, model_name)
        from groq import Groq
        self.client = Groq(api_key=self.api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=self.model_name,
            temperature=0.0, # Set 0 để tăng tính deterministic (ổn định)
        )
        return chat_completion.choices[0].message.content



# ==================== MAIN FUNCTIONS ====================

def build_batch_prompt(papers: list) -> str:
    """Xây dựng prompt cho batch papers. Dùng batch processing để tiết kiệm API cost."""
    input_text = ""

    for paper in papers:
        title = paper.get("title", "N/A")
        abstract = paper.get("abstract", "N/A")
        # Truncate abstract to 400 chars
        if isinstance(abstract, str) and len(abstract) > 400:
            abstract = abstract[:400]
        venue = paper.get("venue", "N/A")

        input_text += f"""
Paper ID: {paper.get("paper_id")}
Title: {title}
Venue: {venue}
Abstract: {abstract}

"""

    return f"""
{system_prompt}

Now classify the following papers.

Return a JSON ARRAY like this:
[
  {{"paper_id": "...", "is_nlp": true, "confidence": 0.9}},
  {{"paper_id": "...", "is_nlp": false, "confidence": 0.8}}
]

Papers:
{input_text}
"""


def parse_batch_response(response: str) -> list:
    """Parse JSON ARRAY response từ LLM. Dùng cho batch processing."""
    results = []
    try:
        # Tìm JSON array trong response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            results_array = json.loads(match.group())
            if isinstance(results_array, list):
                for item in results_array:
                    is_nlp = item.get("is_nlp", False)
                    confidence = float(item.get("confidence", 0.0))
                    
                    # FIX: Nếu confidence=0.0, điều chỉnh thành 0.5
                    if confidence == 0.0:
                        confidence = 0.5
                    
                    results.append({
                        "paper_id": item.get("paper_id"),
                        "is_nlp": is_nlp,
                        "confidence": confidence
                    })
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        pass
    
    return results


def evaluate_batch(evaluator_name: str, evaluator: BaseEvaluator, papers: list) -> list:
    """Gọi evaluator cho batch papers với batch processing (tiết kiệm API cost)."""
    results = []
    
    try:
        # Xây dựng batch prompt cho tất cả papers
        batch_prompt = build_batch_prompt(papers)
        
        # Gọi API 1 lần cho cả batch
        response = evaluator.generate(system_prompt="", user_prompt=batch_prompt)
        
        # Parse batch response (JSON array)
        parsed_results = parse_batch_response(response)
        
        if not parsed_results:
            # Parse thất bại - log error
            print(f"  ⚠️  Parse failed for batch, invalid response: {response[:200]}")
            for paper in papers:
                results.append({
                    "paper_id": paper.get("paper_id"),
                    f"{evaluator_name}_is_nlp": None,
                    f"{evaluator_name}_confidence": None
                })
        else:
            # Parse thành công - format output
            for parsed in parsed_results:
                results.append({
                    "paper_id": parsed.get("paper_id"),
                    f"{evaluator_name}_is_nlp": parsed.get("is_nlp"),
                    f"{evaluator_name}_confidence": parsed.get("confidence")
                })
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"  ❌ {evaluator_name} error for batch: {error_msg}")
        for paper in papers:
            results.append({
                "paper_id": paper.get("paper_id"),
                f"{evaluator_name}_is_nlp": None,
                f"{evaluator_name}_confidence": None
            })
    
    return results


def append_to_csv(output_path: str, rows: list, all_fieldnames: list = None):
    """Append rows vào CSV file, merge với existing data."""
    file_exists = os.path.exists(output_path)
    
    if file_exists:
        # Read existing data
        existing_data = {}
        existing_fieldnames = []
        with open(output_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            for row in reader:
                existing_data[row['paper_id']] = row
        
        # Merge new data with existing
        for new_row in rows:
            pid = new_row['paper_id']
            if pid in existing_data:
                existing_data[pid].update(new_row)
            else:
                existing_data[pid] = new_row
        
        # Get all fieldnames: giữ lại fieldnames cũ + thêm fieldnames mới
        if all_fieldnames is None:
            all_fieldnames = existing_fieldnames.copy() if existing_fieldnames else ['paper_id']
            # Thêm fieldnames mới từ batch_results
            for row in rows:
                for key in row.keys():
                    if key not in all_fieldnames:
                        all_fieldnames.append(key)
        
        # Write updated data
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames, restval='')
            writer.writeheader()
            for row in existing_data.values():
                writer.writerow(row)
    else:
        # Create new file
        if all_fieldnames is None:
            all_fieldnames = ['paper_id'] + [k for k in rows[0].keys() if k != 'paper_id']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def process_papers(
    input_path: str, 
    output_path: str,
    evaluator_name: str,
    evaluator: BaseEvaluator,
    batch_size: int = 25,
    start_from: int = 0
):
    """Xử lý papers theo batch, gọi 1 evaluator, append CSV."""
    
    print(f"\n📂 Input: {input_path}")
    print(f"📊 Output: {output_path}")
    print(f"🤖 Evaluator: {evaluator_name}")
    print(f"📦 Batch size: {batch_size}\n")
    
    # Load papers
    with open(input_path, 'r', encoding='utf-8') as f:
        papers = [json.loads(line) for line in f if line.strip()]
    
    papers = papers[start_from:]
    
    print(f"✅ Loaded {len(papers)} papers\n")
    
    # Process in batches
    total_batches = (len(papers) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(papers))
        batch = papers[start_idx:end_idx]
        
        print(f"⏳ Batch {batch_idx + 1}/{total_batches} ({len(batch)} papers)...")
        
        # Process batch
        batch_results = evaluate_batch(evaluator_name, evaluator, batch)
        
        # Get all fieldnames: include existing fieldnames from CSV + new fields
        all_fieldnames = ['paper_id']
        
        # If CSV exists, read existing fieldnames
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        all_fieldnames = list(reader.fieldnames)
            except Exception:
                pass
        
        # Add new fieldnames from batch_results
        for row in batch_results:
            for key in row.keys():
                if key not in all_fieldnames:
                    all_fieldnames.append(key)
        
        # Append to CSV
        append_to_csv(output_path, batch_results, all_fieldnames)
        
        print(f"✅ Batch {batch_idx + 1} saved\n")
    
    print(f"✅ All batches completed! Results → {output_path}")


def main():
    load_dotenv()
    
    # Initialize Vertex AI with credentials from .env
    gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    gcp_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")  # Default to us-central1
    
    if gcp_project:
        try:
            vertexai.init(project=gcp_project, location=gcp_location)
            print(f"✅ Vertex AI initialized (project: {gcp_project}, location: {gcp_location})\n")
        except Exception as e:
            print(f"⚠️  Vertex AI initialization warning: {e}\n")
    else:
        print("⚠️  GOOGLE_CLOUD_PROJECT not set in .env, skipping Vertex AI init\n")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(base_dir, "data_processed", "final_cleaned_data.jsonl")
    output_path = os.path.join(base_dir, "src", "eval", "eval_results.csv")
    
    # Get API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    # Initialize evaluator - UNCOMMENT ONLY ONE OPTION BELOW
    
    # Option 1: Google Gemini / Vertex AI
    if gemini_key:
        try:
            evaluator = GeminiEvaluator(gemini_key)
            evaluator_name = "gemini"
            print("✅ Gemini initialized\n")
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
            return
    # Option 2: Groq LLaMA (uncomment to use Groq instead)
    # elif groq_key:
    #     try:
    #         evaluator = GroqEvaluator(groq_key)
    #         evaluator_name = "groq"
    #         print("✅ Groq initialized\n")
    #     except Exception as e:
    #         print(f"❌ Groq initialization failed: {e}")
    #         return
    else:
        print("❌ Neither GEMINI_API_KEY nor GROQ_API_KEY found in .env!")
        return
    
    # Process papers
    try:
        process_papers(
            input_path=input_path,
            output_path=output_path,
            evaluator_name=evaluator_name,
            evaluator=evaluator,
            batch_size=25,
            start_from=0
        )
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Progress saved.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

