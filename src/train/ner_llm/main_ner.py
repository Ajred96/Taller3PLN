"""
main_ner.py
===========
Orquesta el pipeline completo de NER con LLMs:
    1. Entrena BETO sobre dataset próstata
    2. Entrena XLM-RoBERTa sobre dataset próstata
    3. Genera tabla comparativa final entre ambos modelos

Uso:
    python -m src.train.ner_llm.main_ner
    python -m src.train.ner_llm.main_ner --modelo beto
    python -m src.train.ner_llm.main_ner --modelo xlmroberta
"""

import argparse
import json
from pathlib import Path

REPORTS_DIR = Path("outputs/reports/ner_llm")


def comparar_modelos() -> None:
    """
    Lee los reportes JSON de BETO y XLM-RoBERTa y genera
    una tabla comparativa final por batch size.
    """
    print("\n" + "=" * 65)
    print("  COMPARACIÓN FINAL: BETO vs XLM-RoBERTa — NER PRÓSTATA")
    print("=" * 65)
    print(f"{'Modelo':<20} {'Batch':>6} {'F1':>8} {'Precision':>12} {'Recall':>10}")
    print("-" * 65)

    for modelo_id, prefijo in [("BETO", "beto"), ("XLM-RoBERTa", "xlmroberta")]:
        resumen_path = REPORTS_DIR / f"{prefijo}_resumen.json"
        if not resumen_path.exists():
            print(f"  {modelo_id}: reporte no encontrado en {resumen_path}")
            continue

        with open(resumen_path) as f:
            resultados = json.load(f)

        for config, m in resultados.items():
            bs  = config.replace("bs", "")
            f1  = m.get("eval_f1", 0)
            pre = m.get("eval_precision", 0)
            rec = m.get("eval_recall", 0)
            print(f"{modelo_id:<20} {bs:>6} {f1:>8.4f} {pre:>12.4f} {rec:>10.4f}")

    print("=" * 65)
    print("\nNota: F1 calculado con seqeval a nivel de entidad (estándar NER).")
    print("      Referencia baseline: bert-base-cased sobre CoNLL2002 → F1 = 0.75")


def main():
    parser = argparse.ArgumentParser(description="Pipeline NER con LLMs sobre dataset próstata")
    parser.add_argument(
        "--modelo",
        choices=["beto", "xlmroberta", "ambos"],
        default="ambos",
        help="Modelo a entrenar (default: ambos)",
    )
    args = parser.parse_args()

    if args.modelo in ("beto", "ambos"):
        print("\n" + "█" * 55)
        print("  FASE 1: BETO NER")
        print("█" * 55)
        from train_beto_ner import main as main_beto
        main_beto()

    if args.modelo in ("xlmroberta", "ambos"):
        print("\n" + "█" * 55)
        print("  FASE 2: XLM-RoBERTa NER")
        print("█" * 55)
        from train_xlmroberta_ner import main as main_xlm
        main_xlm()

    if args.modelo == "ambos":
        comparar_modelos()


if __name__ == "__main__":
    main()
