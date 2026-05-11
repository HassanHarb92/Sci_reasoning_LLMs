"""
Generate a Supporting Information table showing 10 randomly selected ChemBench questions
across all philosopher agents, base models, and GPT architectures.

Usage:
    python generate_si_table.py --data_dir /path/to/csv/files --output si_table.csv

All CSV files should follow the naming convention:
    {Philosopher}_{model}.csv   (e.g., Socrates_gpt5.csv)
    base_{model}.csv            (e.g., base_gpt5.csv)

Output columns:
    Question | Philosopher | Model | Model Answer | Ground Truth | Correct?
"""

import pandas as pd
import ast
import os
import argparse
import random


def extract_question_text(prompt_text_sent):
    """Extract the actual chemistry question from the prompt_text_sent field."""
    try:
        parsed = ast.literal_eval(prompt_text_sent)
        content = parsed[0]['messages'][0]['content']
        # Extract just the question part (between "Question: " and "\n\nYou MUST")
        if 'Question: ' in content:
            q = content.split('Question: ', 1)[1]
            if '\n\nYou MUST' in q:
                q = q.split('\n\nYou MUST')[0]
            elif '\nYou MUST' in q:
                q = q.split('\nYou MUST')[0]
            return q.strip()
        return content[:300].strip()
    except Exception:
        return "N/A"


def parse_filename(filename):
    """Extract philosopher and model from filename like 'Socrates_gpt5.csv'."""
    name = os.path.splitext(filename)[0]
    parts = name.split('_', 1)
    if len(parts) == 2:
        philosopher = parts[0]
        model = parts[1]
        # Clean up model names for readability
        model_map = {
            'gpt4olatest': 'GPT-4o',
            'gpt5': 'GPT-5',
            'gpt51': 'GPT-5.1',
        }
        model_clean = model_map.get(model, model)
        # Base models have no philosopher
        if philosopher.lower() == 'base':
            return 'N/A', model_clean
        return philosopher, model_clean
    return name, 'Unknown'


def main():
    parser = argparse.ArgumentParser(description='Generate SI table for ChemBench results')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory containing all CSV files (default: current dir)')
    parser.add_argument('--output', type=str, default='si_table.csv',
                        help='Output CSV filename')
    parser.add_argument('--n_questions', type=int, default=10,
                        help='Number of random questions to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)

    # Find all CSV files in the directory
    csv_files = sorted([f for f in os.listdir(args.data_dir)
                        if f.endswith('.csv') and f != 'chembench_mae_questions.csv'])

    if not csv_files:
        print(f"No CSV files found in {args.data_dir}")
        return

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {f}")

    # Step 1: Load one file to get the shared UUIDs and select random questions
    ref_file = csv_files[0]
    ref_df = pd.read_csv(os.path.join(args.data_dir, ref_file))
    all_uuids = ref_df['uuid'].tolist()

    # Select N random UUIDs (these will be the same across all files)
    selected_uuids = random.sample(all_uuids, min(args.n_questions, len(all_uuids)))
    print(f"\nSelected {len(selected_uuids)} random questions (seed={args.seed})")

    # Step 2: Process each CSV file and extract data for the selected questions
    rows = []

    for csv_file in csv_files:
        philosopher, model = parse_filename(csv_file)
        filepath = os.path.join(args.data_dir, csv_file)
        df = pd.read_csv(filepath)

        # Filter to selected UUIDs
        df_sel = df[df['uuid'].isin(selected_uuids)]

        for _, row in df_sel.iterrows():
            question_text = extract_question_text(row['prompt_text_sent'])
            ground_truth = row['numeric_target']
            model_answer = row['numeric_model']
            is_correct = row['numeric_is_correct']

            # Clean up correct field
            if pd.isna(is_correct):
                correct_str = 'N/A'
            else:
                correct_str = 'Yes' if int(is_correct) == 1 else 'No'

            rows.append({
                'Question': question_text,
                'Philosopher': philosopher,
                'Model': model,
                'Model Answer': round(model_answer, 2) if pd.notna(model_answer) else 'N/A',
                'Ground Truth': ground_truth,
                'Correct?': correct_str,
                'uuid': row['uuid'],  # keep for sorting, drop later
            })

    # Step 3: Build the final DataFrame
    result_df = pd.DataFrame(rows)

    # Sort by question (uuid), then philosopher, then model
    result_df = result_df.sort_values(['uuid', 'Philosopher', 'Model'])
    result_df = result_df.drop(columns=['uuid'])
    result_df = result_df.reset_index(drop=True)

    # Step 4: Save
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved table with {len(result_df)} rows to {args.output}")
    print(f"({len(selected_uuids)} questions x {len(csv_files)} file combinations)")

    # Also print a preview
    print("\n--- Preview (first 20 rows) ---")
    print(result_df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
