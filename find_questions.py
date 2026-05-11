"""
Find discriminating ChemBench questions for each Philosophy Agent.

For each philosopher and each model, identify questions where:
  - The named philosopher got it CORRECT
  - The base model got it WRONG
  - Ideally, most OTHER philosophers also got it wrong (high discrimination)

This helps surface real examples where a specific agent's reasoning style
genuinely made a difference, rather than easy questions everyone solves.

Usage:
    python find_discriminating_questions.py
    (assumes CSVs in current directory)
"""

import pandas as pd
import ast
import os
from collections import defaultdict


PHILOSOPHERS = ['Aristotle', 'Descartes', 'Hegel', 'Hume', 'Kant', 'Plato', 'Socrates']
MODELS = {'gpt4olatest': 'GPT-4o', 'gpt5': 'GPT-5', 'gpt51': 'GPT-5.1'}


def extract_question_text(prompt_text_sent):
    try:
        parsed = ast.literal_eval(prompt_text_sent)
        content = parsed[0]['messages'][0]['content']
        if 'Question: ' in content:
            q = content.split('Question: ', 1)[1]
            for marker in ['\n\nYou MUST', '\nYou MUST']:
                if marker in q:
                    q = q.split(marker)[0]
                    break
            return q.strip()
        return content[:300].strip()
    except Exception:
        return "N/A"


def load_csv(data_dir, philosopher, model_key):
    """Load a CSV. Philosopher can be 'base' for base models."""
    filename = f"{philosopher}_{model_key}.csv"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Keep the columns we need
    return df[['uuid', 'prompt_text_sent', 'numeric_target', 'numeric_model',
               'numeric_is_correct']].copy()


def main():
    data_dir = '.'

    # Load all files into a nested dict: data[model_key][philosopher] = df
    data = {}
    for model_key in MODELS:
        data[model_key] = {}
        base_df = load_csv(data_dir, 'base', model_key)
        if base_df is not None:
            data[model_key]['base'] = base_df
        for phil in PHILOSOPHERS:
            df = load_csv(data_dir, phil, model_key)
            if df is not None:
                data[model_key][phil] = df

    # Report what was found
    print("Loaded data:")
    for model_key, agents in data.items():
        print(f"  {MODELS[model_key]}: {list(agents.keys())}")

    # For each (philosopher, model), find questions where:
    #   - This philosopher is CORRECT (numeric_is_correct == 1)
    #   - Base is WRONG
    # Then rank by how few OTHER philosophers also got it right (more unique = more discriminating)

    print("\n" + "=" * 80)
    print("DISCRIMINATING QUESTIONS FOR EACH PHILOSOPHER")
    print("=" * 80)

    results = defaultdict(list)

    for model_key, agents in data.items():
        model_name = MODELS[model_key]
        if 'base' not in agents:
            continue
        base_df = agents['base'].set_index('uuid')

        for phil in PHILOSOPHERS:
            if phil not in agents:
                continue
            phil_df = agents[phil].set_index('uuid')

            # Find UUIDs where this phil is correct AND base is wrong
            phil_correct = set(phil_df[phil_df['numeric_is_correct'] == 1].index)
            base_wrong = set(base_df[base_df['numeric_is_correct'] != 1].index)
            candidates = phil_correct & base_wrong

            # For each candidate, count how many OTHER philosophers also got it right
            for uuid in candidates:
                other_correct_count = 0
                for other_phil in PHILOSOPHERS:
                    if other_phil == phil or other_phil not in agents:
                        continue
                    other_df = agents[other_phil]
                    row = other_df[other_df['uuid'] == uuid]
                    if not row.empty and row.iloc[0]['numeric_is_correct'] == 1:
                        other_correct_count += 1

                q_text = extract_question_text(phil_df.loc[uuid, 'prompt_text_sent'])
                target = phil_df.loc[uuid, 'numeric_target']
                phil_answer = phil_df.loc[uuid, 'numeric_model']
                base_answer = base_df.loc[uuid, 'numeric_model']

                results[phil].append({
                    'model': model_name,
                    'uuid': uuid,
                    'question': q_text,
                    'target': target,
                    'phil_answer': phil_answer,
                    'base_answer': base_answer,
                    'others_correct': other_correct_count,
                    'uniqueness_score': 6 - other_correct_count,  # higher = more unique
                })

    # Print top 5 most discriminating questions for each philosopher
    for phil in PHILOSOPHERS:
        print(f"\n\n{'=' * 80}")
        print(f"TOP DISCRIMINATING QUESTIONS FOR: {phil}")
        print(f"(where {phil} succeeded, base failed, fewer other agents succeeded)")
        print("=" * 80)

        sorted_results = sorted(results[phil], key=lambda x: (-x['uniqueness_score'], x['model']))

        if not sorted_results:
            print(f"  No discriminating questions found for {phil}")
            continue

        for i, r in enumerate(sorted_results[:5]):
            print(f"\n  [{i+1}] Model: {r['model']} | Unique among 6 others: {r['uniqueness_score']}/6 failed")
            print(f"      Question: {r['question'][:200]}")
            print(f"      Target: {r['target']} | {phil}: {r['phil_answer']} | Base: {r['base_answer']}")


if __name__ == '__main__':
    main()
