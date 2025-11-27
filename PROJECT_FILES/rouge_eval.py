import pandas as pd
import evaluate

# Load gold + generated SOAPs
df = pd.read_csv("SOAP_FINAL/preds_clean.csv", encoding="utf-8")

# sanity checks
assert "reference" in df.columns, "preds_clean.csv must contain a 'reference' column"
assert "prediction" in df.columns, "preds_clean.csv must contain a 'prediction' column"

refs = df["reference"].astype(str).tolist()
preds = df["prediction"].astype(str).tolist()

# Load ROUGE scorer
rouge = evaluate.load("rouge")

scores = rouge.compute(
    predictions=preds,
    references=refs,
    use_aggregator=True
)

print("\n========== ROUGE Evaluation ==========\n")
for k, v in scores.items():
    print(f"{k}: {v:.4f}")

print("\n======================================\n")
