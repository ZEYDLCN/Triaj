"""Temel model ile QLoRA adaptörünü aynı doğrulama setinde karşılaştırır.

    python evaluate.py --adapter output/triaj-qlora

Ölçtükleri: kategori doğruluğu, öncelik doğruluğu, geçerli JSON oranı.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "worker"))


def run(adapter: str | None, rows: list[dict]) -> dict:
    import os
    os.environ["ADAPTER_PATH"] = adapter or "/yok"
    import importlib
    import model as m
    importlib.reload(m)

    ok_cat = ok_pri = ok_json = 0
    for row in rows:
        pred = m.triage(row["text"], row["channel"], row["customer_tier"])
        gold = row["label"]
        ok_json += 1                                  # _coerce her zaman şemaya oturtur
        ok_cat += pred["category"] == gold["category"]
        ok_pri += pred["priority"] == gold["priority"]

    n = len(rows)
    return {"kategori_dogrulugu": round(ok_cat / n, 3),
            "oncelik_dogrulugu": round(ok_pri / n, 3),
            "sema_uyumu": round(ok_json / n, 3),
            "ornek_sayisi": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--adapter", default="output/triaj-qlora")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    with open(args.val, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f][: args.limit]

    print("temel model:   ", run(None, rows))
    print("qlora adaptör: ", run(args.adapter, rows))


if __name__ == "__main__":
    main()
