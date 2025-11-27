import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# Import generate_one + helpers from infer.py
from infer import generate_one

def load_csv_safe(path):
    """Try UTF-8 first, fallback to ISO-8859-1 if decoding fails."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="ISO-8859-1")

def main(args):
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)

    print(f"Reading input CSV: {args.input_csv}")
    df = load_csv_safe(args.input_csv)

    # ✅ Ensure dialogue column exists
    if "dialogue" not in df.columns:
        raise ValueError(
            "CSV must contain a column named 'dialogue'. "
            f"Columns found: {list(df.columns)}"
        )

    df["dialogue"] = df["dialogue"].astype(str)

    preds = []
    print("\nGenerating SOAP notes...\n")
    for text in tqdm(df["dialogue"]):
        preds.append(
            generate_one(
                text,
                model,
                tokenizer,
                max_input=args.max_input,
                max_out=args.max_out,
                device=device
            )
        )

    df["generated_soap"] = preds

    output_csv = args.output_csv or "preds.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\n✅ DONE — Results saved to: {output_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--max_input", type=int, default=2048)
    parser.add_argument("--max_out", type=int, default=450)
    parser.add_argument("--use_cuda", action="store_true")

    args = parser.parse_args()
    main(args)
