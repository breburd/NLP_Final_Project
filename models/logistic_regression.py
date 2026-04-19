import argparse
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, get_scores, load_data, make_folder, save_json, split_data


def build_model(max_features):
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "logistic_regression"))
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.data_path)
    train_df, valid_df, test_df = split_data(df, seed=args.seed)

    model = build_model(args.max_features)
    model.fit(train_df["text"], train_df["label"])

    valid_pred = model.predict(valid_df["text"])
    test_pred = model.predict(test_df["text"])

    results = {
        "baseline": "logistic_regression",
        "num_train": len(train_df),
        "num_valid": len(valid_df),
        "num_test": len(test_df),
        "max_features": args.max_features,
        "valid": get_scores(valid_df["label"], valid_pred),
        "test": get_scores(test_df["label"], test_pred),
    }

    output_dir = make_folder(args.output_dir)
    save_json(results, output_dir / "metrics.json")

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("logistic regression finished")
    print(output_dir / "metrics.json")


if __name__ == "__main__":
    main()
