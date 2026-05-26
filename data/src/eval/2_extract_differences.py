import pandas as pd

# Read the evaluation results
df = pd.read_csv('eval_results.csv')

# Filter rows where groq_is_nlp != gemini_is_nlp
differences = df[df['groq_is_nlp'] != df['gemini_is_nlp']]

# Select only required columns
result_df = differences[['paper_id', 'groq_is_nlp', 'gemini_is_nlp']]

# Save to CSV
result_df.to_csv('differences.csv', index=False)

print(f"Total papers with different results: {len(result_df)}")
print(f"\nSaved to differences.csv")
print(f"\nFirst few rows:")
print(result_df.head(10))
