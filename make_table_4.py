#!/usr/bin/env python3
"""
Generate representative ChemBench questions for Table S4.

Input:
    chembench_mae_questions.csv

Output:
    table_s4_representative_questions.csv
    table_s4_representative_questions.md

Usage:
    python generate_table_s4.py
"""

import ast
import re
import pandas as pd
from pathlib import Path


INPUT_CSV = "chembench_mae_questions.csv"
OUTPUT_CSV = "table_s4_representative_questions.csv"
OUTPUT_MD = "table_s4_representative_questions.md"

# Number of representative examples to include
N_EXAMPLES = 12


DOMAIN_KEYWORDS = {
    "physical-chemistry": "Physical chemistry",
    "analytical-chemistry": "Analytical chemistry",
    "general-chemistry": "General chemistry",
    "inorganic-chemistry": "Inorganic chemistry",
    "organic-chemistry": "Organic chemistry",
    "technical-chemistry": "Technical chemistry",
    "materials-science": "Materials science",
    "toxicity-safety": "Toxicity/safety",
    "computational-chemistry": "Computational chemistry",
}


TASK_KEYWORDS = {
    "requires-calculation": "Calculation",
    "requires-reasoning": "Reasoning",
    "requires-knowledge": "Knowledge",
    "requires-intuition": "Intuition",
}


def safe_literal_eval(value):
    """Safely parse Python-like strings from the CSV."""
    if pd.isna(value):
        return None

    if not isinstance(value, str):
        return value

    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def extract_keywords(value):
    """Parse the keywords column into a list."""
    parsed = safe_literal_eval(value)
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed]
    return []


def extract_target(value):
    """Parse the targets column and return the first target value."""
    parsed = safe_literal_eval(value)

    if isinstance(parsed, list) and len(parsed) > 0:
        return str(parsed[0])

    if isinstance(value, str):
        match = re.search(r"\['([^']+)'\]", value)
        if match:
            return match.group(1)

    return ""


def extract_question(prompt_value):
    """
    Extract the question text from the prompts.batch column.

    Expected format:
    [{'messages': [{'role': 'user', 'content': "... Question: ... You MUST ..."}]}]
    """
    parsed = safe_literal_eval(prompt_value)

    content = ""

    try:
        content = parsed[0]["messages"][0]["content"]
    except Exception:
        if isinstance(prompt_value, str):
            content = prompt_value

    if not content:
        return ""

    # Extract text between "Question:" and "You MUST"
    match = re.search(
        r"Question:\s*(.*?)\s*You MUST include the final answer",
        content,
        flags=re.DOTALL,
    )

    if match:
        question = match.group(1).strip()
    else:
        question = content.strip()

    # Clean whitespace
    question = re.sub(r"\s+", " ", question)

    return question


def infer_domain(keywords):
    """Infer chemistry subdomain from keywords."""
    for key, label in DOMAIN_KEYWORDS.items():
        if key in keywords:
            return label
    return "Other"


def infer_task_type(keywords):
    """Infer task type from keywords."""
    task_types = []

    for key, label in TASK_KEYWORDS.items():
        if key in keywords:
            task_types.append(label)

    if task_types:
        return " + ".join(task_types)

    return "Unspecified"


def infer_difficulty(keywords):
    """Infer difficulty from keywords."""
    for kw in keywords:
        if kw.startswith("difficulty-"):
            return kw.replace("difficulty-", "").capitalize()
    return "Unspecified"


def build_clean_table(df):
    """Build cleaned Table S4 candidate rows."""
    rows = []

    for _, row in df.iterrows():
        keywords = extract_keywords(row.get("keywords", ""))
        question = extract_question(row.get("prompts.batch", ""))
        target = extract_target(row.get("targets", ""))

        rows.append(
            {
                "Name": row.get("name", ""),
                "UUID": row.get("uuid", ""),
                "Description": row.get("description", ""),
                "Subdomain": infer_domain(keywords),
                "Task type": infer_task_type(keywords),
                "Difficulty": infer_difficulty(keywords),
                "Question": question,
                "Reference answer": target,
                "Representative base prediction": "",
                "Representative PA prediction": "",
                "Notes": "",
            }
        )

    return pd.DataFrame(rows)


def select_representative_examples(clean_df, n_examples=12):
    """
    Select a balanced set of examples across subdomains and task types.

    Priority:
    1. Keep common manuscript subdomains first.
    2. Include calculation, reasoning, and knowledge examples.
    3. Avoid duplicate domains when possible.
    """
    preferred_domains = [
        "Physical chemistry",
        "Analytical chemistry",
        "General chemistry",
        "Inorganic chemistry",
        "Organic chemistry",
        "Computational chemistry",
        "Materials science",
        "Technical chemistry",
        "Toxicity/safety",
    ]

    preferred_tasks = [
        "Calculation",
        "Reasoning",
        "Knowledge",
        "Reasoning + Knowledge",
        "Knowledge + Calculation",
        "Reasoning + Calculation",
    ]

    selected = []
    used_indices = set()

    # First pass: one per preferred domain
    for domain in preferred_domains:
        subset = clean_df[clean_df["Subdomain"] == domain]

        if subset.empty:
            continue

        # Prefer rows with clear task labels
        subset = subset.sort_values(
            by=["Task type", "Difficulty", "Name"],
            key=lambda col: col.astype(str),
        )

        idx = subset.index[0]
        selected.append(idx)
        used_indices.add(idx)

        if len(selected) >= n_examples:
            break

    # Second pass: ensure task-type diversity
    for task in preferred_tasks:
        subset = clean_df[
            clean_df["Task type"].str.contains(task, regex=False, na=False)
            & ~clean_df.index.isin(used_indices)
        ]

        if subset.empty:
            continue

        idx = subset.index[0]
        selected.append(idx)
        used_indices.add(idx)

        if len(selected) >= n_examples:
            break

    # Third pass: fill remaining rows
    if len(selected) < n_examples:
        remaining = clean_df[~clean_df.index.isin(used_indices)]
        for idx in remaining.index:
            selected.append(idx)
            if len(selected) >= n_examples:
                break

    result = clean_df.loc[selected].reset_index(drop=True)
    result.insert(0, "Example", [f"S4-{i+1}" for i in range(len(result))])

    return result


def save_markdown_table(df, output_path):
    """Save a markdown version for easy copy-paste into SI."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Table S4. Representative ChemBench questions used in this study.\n\n")
        f.write(df.to_markdown(index=False))


def main():
    input_path = Path(INPUT_CSV)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_CSV}")

    df = pd.read_csv(input_path)
    clean_df = build_clean_table(df)
    representative_df = select_representative_examples(clean_df, N_EXAMPLES)

    representative_df.to_csv(OUTPUT_CSV, index=False)
    save_markdown_table(representative_df, OUTPUT_MD)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_MD}")
    print()
    print(representative_df[["Example", "Subdomain", "Task type", "Question", "Reference answer"]])


if __name__ == "__main__":
    main()
