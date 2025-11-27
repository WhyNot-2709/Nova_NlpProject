from bert_score import score
import pandas as pd

df_ref = pd.read_csv("SOAP_FINAL/test.csv", encoding="latin1")
df_pred = pd.read_csv("SOAP_FINAL/preds.csv", encoding="latin1")

references = df_ref["gold_soap"].tolist()
candidates = df_pred["prediction"].tolist()

P, R, F1 = score(
    candidates,
    references,
    model_type="bert-base-uncased",
    batch_size=2,
    idf=False,
    device="cuda"
)

print("BERTScore P:", P.mean().item())
print("BERTScore R:", R.mean().item())
print("BERTScore F1:", F1.mean().item())
