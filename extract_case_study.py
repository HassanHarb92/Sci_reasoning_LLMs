#!/usr/bin/env python3

import ast
import re
from pathlib import Path

import pandas as pd


QUESTION_FILE = "chembench_mae_questions.csv"
CASE_STUDY_NAME = "number_1h_nmr_shifts"

OUTPUT_FILE = "case_study_outputs_clean.csv"
OUTPUT_MD = "case_study_outputs_clean.md"

# Keep the case study focused.
# Add/remove files here as needed.
FILES_TO_USE = [
    "base_gpt4olatest.csv",
    "Socrates_gpt4olatest.csv",
    "Plato_gpt4olatest.csv",
    "Kant_gpt4olatest.csv",
    "base_gpt5.csv",
    "Socrates_gpt5.csv",
    "base_gpt51.csv",
    "Socrates_gpt51.csv",
    "Descartes_gpt51.csv",
]

# Columns that should NEVER be treated as model outputs.
BAD_TEXT_COLUMNS = {
    "prompts.batch",
    "prompt",
    "prompts",
    "messages",
    "targets",
    "keywords",
    "description",
    "name",
    "uuid",
    "preferred_score",
}

# Put likely response columns first.
POSSIBLE_RESPONSE_COLUMNS = [
    "response",
    "model_response",
    "full_response",
    "raw_response",
    "completion",
    "output",
    "model_output",
    "answer",
    "content",
    "result",
]

POSSIBLE_PREDICTION_COLUMNS = [
    "prediction",
    "predicted",
    "parsed_answer",
    "extracted_answer",
    "final_answer",
    "numeric_answer",
]


def safe_eval(x):
    if pd.isna(x):
        return None
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return None


def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def extract_question(prompt_value):
    parsed = safe_eval(prompt_value)

    content = ""
    try:
        content = parsed[0]["messages"][0]["content"]
    except Exception:
        content = str(prompt_value)

    match = re.search(
        r"Question:\s*(.*?)\s*You MUST include the final answer",
        content,
        flags=re.DOTALL,
    )

    if match:
        return clean_text(match.group(1))

    return clean_text(content)


def extract_target(target_value):
    parsed = safe_eval(target_value)
    if isinstance(parsed, list) and parsed:
        return str(parsed[0])
    return str(target_value)


def extract_answer_from_text(text):
    """Extract [ANSWER]value[/ANSWER] if present."""
    if pd.isna(text):
        return ""

    match = re.search(
        r"\[ANSWER\]\s*([^[]+?)\s*\[/ANSWER\]",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return ""


def find_response_column(df):
    """
    Find the column that contains the model's response.
    Avoid prompt/question metadata columns.
    """
    for col in POSSIBLE_RESPONSE_COLUMNS:
        if col in df.columns and col not in BAD_TEXT_COLUMNS:
            return col

    # Fallback: choose the longest text column that is not metadata.
    candidates = []

    for col in df.columns:
        if col in BAD_TEXT_COLUMNS:
            continue

        if df[col].dtype == object:
            avg_len = df[col].astype(str).str.len().mean()
            candidates.append((col, avg_len))

    if not candidates:
        raise ValueError("Could not find a model response column.")

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def find_prediction_column(df):
    for col in POSSIBLE_PREDICTION_COLUMNS:
        if col in df.columns:
            return col
    return None


def load_question_metadata():
    qdf = pd.read_csv(QUESTION_FILE)

    row = qdf[qdf["name"] == CASE_STUDY_NAME]

    if row.empty:
        raise ValueError(f"Could not find question name: {CASE_STUDY_NAME}")

    row = row.iloc[0]

    return {
        "name": row["name"],
        "uuid": row["uuid"],
        "description": row["description"],
        "keywords": row["keywords"],
        "question": extract_question(row["prompts.batch"]),
        "reference_answer": extract_target(row["targets"]),
    }


def parse_filename(file_name):
    stem = Path(file_name).stem
    agent, model = stem.split("_", 1)
    return agent, model


def get_matching_row(df, metadata, file_name):
    if "uuid" in df.columns:
        match = df[df["uuid"].astype(str) == str(metadata["uuid"])]
    elif "name" in df.columns:
        match = df[df["name"].astype(str) == str(metadata["name"])]
    else:
        raise ValueError(f"{file_name} has no uuid or name column.")

    if match.empty:
        raise ValueError(f"No matching row found in {file_name}.")

    return match.iloc[0]


def load_case_outputs(metadata):
    rows = []

    for file_name in FILES_TO_USE:
        path = Path(file_name)

        if not path.exists():
            print(f"Skipping missing file: {file_name}")
            continue

        df = pd.read_csv(path)
        row = get_matching_row(df, metadata, file_name)

        response_col = find_response_column(df)
        prediction_col = find_prediction_column(df)

        response = row[response_col]

        if prediction_col:
            prediction = row[prediction_col]
        else:
            prediction = extract_answer_from_text(response)

        agent, model = parse_filename(file_name)

        rows.append(
            {
                "agent": agent,
                "model": model,
                "file": file_name,
                "response_column": response_col,
                "prediction": prediction,
                "response": response,
            }
        )

    return pd.DataFrame(rows)


def write_markdown(metadata, outputs):
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Text S3. Case Study of Agent Reasoning Behavior\n\n")

        f.write("## Case-study question\n\n")
        f.write(f"**ChemBench ID:** {metadata['name']}\n\n")
        f.write(f"**UUID:** {metadata['uuid']}\n\n")
        f.write(f"**Description:** {metadata['description']}\n\n")
        f.write(f"**Keywords:** {metadata['keywords']}\n\n")
        f.write(f"**Question:** {metadata['question']}\n\n")
        f.write(f"**Reference answer:** {metadata['reference_answer']}\n\n")

        f.write("## Selected model outputs\n\n")

        for _, row in outputs.iterrows():
            f.write(f"### {row['agent']} / {row['model']}\n\n")
            f.write(f"**Prediction:** {row['prediction']}\n\n")
            f.write(f"**Source file:** `{row['file']}`\n\n")
            f.write(f"**Response column:** `{row['response_column']}`\n\n")
            f.write("**Full response:**\n\n")
            f.write(str(row["response"]).strip())
            f.write("\n\n---\n\n")


def main():
    metadata = load_question_metadata()
    outputs = load_case_outputs(metadata)

    if outputs.empty:
        raise ValueError("No outputs were found.")

    agent_order = {
        "base": 0,
        "Socrates": 1,
        "Aristotle": 2,
        "Descartes": 3,
        "Hegel": 4,
        "Hume": 5,
        "Plato": 6,
        "Kant": 7,
    }

    model_order = {
        "gpt4olatest": 0,
        "gpt5": 1,
        "gpt51": 2,
    }

    outputs["agent_order"] = outputs["agent"].map(agent_order).fillna(99)
    outputs["model_order"] = outputs["model"].map(model_order).fillna(99)

    outputs = (
        outputs.sort_values(["model_order", "agent_order"])
        .drop(columns=["model_order", "agent_order"])
        .reset_index(drop=True)
    )

    outputs.to_csv(OUTPUT_FILE, index=False)
    write_markdown(metadata, outputs)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {OUTPUT_MD}")
    print()
    print("Question:")
    print(metadata["question"])
    print()
    print("Reference answer:", metadata["reference_answer"])
    print()
    print(outputs[["model", "agent", "prediction", "response_column", "file"]].to_string(index=False))


if __name__ == "__main__":
    main()
