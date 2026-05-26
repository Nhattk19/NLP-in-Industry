#!/usr/bin/env python3
"""
CLI Entry Point for Paper RAG Agent
Use this to run the agent from command line
"""

import sys
import argparse
import json
import os
from pathlib import Path

# Disable ChromaDB telemetry before any imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agent.agent import get_agent


def _dedupe_by_paper_id(items: list[dict]) -> list[dict]:
    """Keep the first item for each paper_id."""
    unique = []
    seen = set()

    for item in items:
        key = str(item.get("paper_id") or item.get("chunk_id") or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique


def print_result(result: dict):
    """Pretty print agent result"""
    
    print("\n" + "=" * 80)
    print("[RESULT]")
    print("=" * 80)
    
    if not result.get("success"):
        print(f"ERROR: {result.get('error')}")
        return
    
    # Print answer
    print(f"\nAnswer:\n{result.get('answer', 'N/A')}")
    
    # Print sources
    sources = result.get('sources', [])
    if sources:
        sources = _dedupe_by_paper_id(sources)
        print(f"\nSources ({len(sources)}):")
        for i, source in enumerate(sources[:5], 1):  # Show top 5 unique papers
            print(f"  [{i}] {source.get('title')}")
            print(f"      ID: {source.get('paper_id')}")
    
    # Print external papers if used
    external_papers = result.get('external_papers', [])
    if external_papers and result.get('used_external_papers'):
        external_papers = _dedupe_by_paper_id(external_papers)
        print(f"\n[EXT] External Papers (From arXiv):")
        for i, paper in enumerate(external_papers[:5], 1):
            print(f"  [{i}] {paper.get('title')}")
            source_url = paper.get("source_url") or paper.get("url") or ""
            print(f"      Source: {paper.get('source')} | URL: {source_url[:60]}...")
    
    # Print metadata
    print(f"\nMetadata:")
    print(f"  Confidence: {round(result.get('confidence', 0) * 100)}%")
    print(f"  Intent: {result.get('intent')}")
    print(f"  Search Mode: {result.get('search_mode_used')}")
    print(f"  Time: {result.get('execution_time_ms')}ms")
    
    if result.get('external_search_triggered'):
        print(f"  [WARN] External search triggered (answer quality was low)")
    
    if result.get('used_external_papers'):
        print(f"  [OK] Answer was improved with external papers")
    
    feedback = result.get('feedback_info')
    if feedback:
        print(f"\nEvaluation Feedback:")
        print(f"  Answer Quality: {'Good' if feedback.get('answer_is_good') else 'Needs improvement'}")
        print(f"  Reason: {feedback.get('reason')}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="NLP Paper RAG Agent - Ask questions about papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python run_agent.py --query "What is SOTA for NER?"
  
  # Interactive mode
  python run_agent.py --interactive
  
  # Run examples
  python run_agent.py --examples
  
  # Batch queries from file
  python run_agent.py --batch queries.json
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to process"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode (ask multiple queries)"
    )
    
    parser.add_argument(
        "--examples", "-e",
        action="store_true",
        help="Run example queries"
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="Batch file with queries (JSON format)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file to save results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    print("\n[AGENT] Initializing agent...")
    agent = get_agent()
    
    results = []
    
    # Single query
    if args.query:
        print(f"\nProcessing query: {args.query}")
        result = agent.run(args.query)
        results.append(result)
        print_result(result)
    
    # Interactive mode
    elif args.interactive:
        print("\n[MODE] Interactive Mode (type 'quit' to exit)")
        print("-" * 80)
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break
                if not query:
                    continue
                
                result = agent.run(query)
                results.append(result)
                print_result(result)
                
            except KeyboardInterrupt:
                print("\n[EXIT] Exiting...")
                break
    
    # Example queries
    elif args.examples:
        examples = [
            "What is SOTA for Named Entity Recognition?",
            "Explain BERT pretraining",
            "What are transformer attention mechanisms?",
            "What is the difference between RNN and LSTM?",
            "What is zero-shot learning?",
        ]
        
        print(f"\n[MODE] Running {len(examples)} example queries...")
        for query in examples:
            result = agent.run(query)
            results.append(result)
            print_result(result)
    
    # Batch mode
    elif args.batch:
        print(f"\n[MODE] Running batch from {args.batch}")
        
        with open(args.batch, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        queries = data if isinstance(data, list) else data.get("queries", [])
        
        for query in queries:
            if isinstance(query, dict):
                query = query.get("query", query.get("text", ""))
            
            if query:
                result = agent.run(query)
                results.append(result)
                print_result(result)
    
    # Default: interactive mode
    else:
        print("\n[MODE] Interactive Mode (type 'quit' to exit)")
        print("-" * 80)
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break
                if not query:
                    continue
                
                result = agent.run(query)
                results.append(result)
                print_result(result)
                
            except KeyboardInterrupt:
                print("\n[EXIT] Exiting...")
                break
    
    # Save results if output file specified
    if args.output and results:
        print(f"\n[MODE] Saving results to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved {len(results)} results")


if __name__ == "__main__":
    main()
