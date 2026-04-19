import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "preprocess" / "enron_emails_labeled.csv"


def run_one_command(command_list):
    print("")
    print("running:")
    print(" ".join(str(x) for x in command_list))
    result = subprocess.run(command_list, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--run_preprocess", action="store_true")
    parser.add_argument("--run_keyword", action="store_true")
    parser.add_argument("--run_logistic", action="store_true")
    parser.add_argument("--run_bert", action="store_true")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--bert_train_size", type=int, default=30000)
    parser.add_argument("--bert_valid_size", type=int, default=5000)
    parser.add_argument("--bert_test_size", type=int, default=5000)
    args = parser.parse_args()

    want_everything = args.run_all or not any(
        [args.run_preprocess, args.run_keyword, args.run_logistic, args.run_bert]
    )

    if args.run_preprocess or want_everything:
        if args.dataset_path == "":
            print("you need --dataset_path if you want to run preprocess")
            raise SystemExit(1)

        run_one_command(
            [
                sys.executable,
                "preprocess/preprocess.py",
                "--dataset_path",
                args.dataset_path,
                "--output_path",
                str(DEFAULT_DATA_PATH),
            ]
        )

    if args.run_keyword or want_everything:
        run_one_command([sys.executable, "models/keyword_filter.py", "--data_path", args.data_path])

    if args.run_logistic or want_everything:
        run_one_command([sys.executable, "models/logistic_regression.py", "--data_path", args.data_path])

    if args.run_bert or want_everything:
        run_one_command(
            [
                sys.executable,
                "models/bert_baseline.py",
                "--data_path",
                args.data_path,
                "--train_size",
                str(args.bert_train_size),
                "--valid_size",
                str(args.bert_valid_size),
                "--test_size",
                str(args.bert_test_size),
            ]
        )

    print("")
    print("done")


if __name__ == "__main__":
    main()
