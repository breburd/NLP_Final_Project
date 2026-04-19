import argparse
import re

from common import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, get_scores, load_data, make_folder, save_json, split_data


KEYWORDS = [
    "attorney-client",
    "privileged",
    "confidential",
    "legal advice",
    "counsel",
    "litigation",
]


def run_keyword_model(text_series):
    pattern = "|".join(re.escape(word) for word in KEYWORDS)
    matches = text_series.str.lower().str.contains(pattern, regex=True, na=False)
    return matches.astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "keyword_filter"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.data_path)
    train_df, valid_df, test_df = split_data(df, seed=args.seed)

    valid_pred = run_keyword_model(valid_df["text"])
    test_pred = run_keyword_model(test_df["text"])

    results = {
        "baseline": "keyword_filter",
        "keywords": KEYWORDS,
        "num_train": len(train_df),
        "num_valid": len(valid_df),
        "num_test": len(test_df),
        "valid": get_scores(valid_df["label"], valid_pred),
        "test": get_scores(test_df["label"], test_pred),
    }

    output_dir = make_folder(args.output_dir)
    save_json(results, output_dir / "metrics.json")

    sample_rows = test_df[["subject", "label"]].head(20).copy()
    sample_rows["prediction"] = test_pred.head(20).tolist()
    sample_rows.to_json(output_dir / "sample_predictions.json", orient="records", indent=2)

    print("keyword baseline finished")
    print(output_dir / "metrics.json")


if __name__ == "__main__":
    main()
