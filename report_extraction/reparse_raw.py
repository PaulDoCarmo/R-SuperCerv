#!/usr/bin/env python3
"""Re-parse les reponses LLM DEJA sauvegardees (colonne 'DNN Answer') avec le parser corrige,
SANS re-inferer le LLM. Reproduit exactement le format de run_inference (1 ligne/lesion,
ou Lesion Index 0 + 'No lesions mentioned' si aucune)."""
import argparse, csv, os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "LLM_inference"))
from LLM_inference import parse_answer, get_prompt_schema   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ancien raw CSV (avec colonne DNN Answer)")
    ap.add_argument("--output", required=True, help="nouveau raw CSV re-parse")
    ap.add_argument("--prompt_id", type=int, default=5)
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    schema = get_prompt_schema(a.prompt_id)
    out_cols = ["ID", "Lesion Index", *schema["output_fields"], "DNN Answer", "Report"]
    per_id = df.groupby("ID", sort=False).first().reset_index()   # 1 reponse (DNN Answer) / ID

    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    n_old = len(df); n_new = 0
    with open(a.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols); w.writeheader()
        for _, row in per_id.iterrows():
            ans = str(row["DNN Answer"]); rep = str(row.get("Report", ""))
            lesions = parse_answer(ans, a.prompt_id)
            if not lesions:
                empty = {f: "U" for f in schema["output_fields"]}; empty["type"] = "No lesions mentioned"
                w.writerow({"ID": row["ID"], "Lesion Index": 0, **empty,
                            "DNN Answer": ans, "Report": rep})
            else:
                n_new += len(lesions)
                for i, les in enumerate(lesions, 1):
                    w.writerow({"ID": row["ID"], "Lesion Index": i, **les,
                                "DNN Answer": ans, "Report": rep})
    print(f"reports re-parses : {len(per_id)} | lignes-lesions old={n_old} -> new={n_new}")
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
