"""
train_xlmroberta_ner.py
=======================
Fine-tuning de XLM-RoBERTa (xlm-roberta-large)
sobre el dataset biomédico de próstata para NER.

Prueba batch sizes 8, 16 y 32 con Early Stopping.
Sube el mejor modelo a Hugging Face.

Nota sobre XLM-RoBERTa vs BETO:
- XLM-RoBERTa es más grande (560M params vs 110M de BETO)
- Fue entrenado en 100 idiomas incluyendo español
- Generalmente supera a BETO en tareas con vocabulario especializado
- Requiere más memoria GPU: usar batch_size=8 si la GPU tiene < 16 GB

Uso:
    python -m src.train.ner_llm.train_xlmroberta_ner
"""

import os
import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from huggingface_hub import login

from utils_ner import (
    cargar_dataset,
    tokenizar_y_alinear,
    compute_metrics,
    mostrar_estadisticas,
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
)

# ── Configuración ─────────────────────────────────────────────────────────────

MODELO_BASE  = "xlm-roberta-large"
DATA_DIR     = Path("data/processed/prostata_json")
OUTPUT_BASE  = Path("outputs/models/xlmroberta_ner")
REPORTS_DIR  = Path("outputs/reports/ner_llm")
HF_REPO_BASE = "xlmroberta-ner-prostata"

BATCH_SIZES          = [8, 16, 32]
NUM_EPOCHS           = 10
MAX_LENGTH           = 512
LEARNING_RATE        = 1e-5     # XLM-RoBERTa es más sensible: lr más baja que BETO
WEIGHT_DECAY         = 0.01
EARLY_STOP_PATIENCE  = 3
GRADIENT_ACCUMULATION = 2       # acumula gradientes para simular batch más grande en GPU pequeña

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Autenticación ─────────────────────────────────────────────────────────────

def autenticar_hf():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠ No se encontró HF_TOKEN en variables de entorno.")
        token = input("Ingresa tu token de Hugging Face (Write): ").strip()
    login(token=token)
    print("✔ Autenticado en Hugging Face")


# ── Pipeline de entrenamiento ─────────────────────────────────────────────────

def entrenar_con_batch(
    dataset_tokenizado: dict,
    tokenizer,
    batch_size: int,
    output_dir: Path,
    hf_repo: str,
) -> dict:
    """
    Entrena XLM-RoBERTa con un batch size específico.

    XLM-RoBERTa usa un tokenizador SentencePiece en lugar de WordPiece,
    por lo que el alineamiento de etiquetas funciona igual pero los
    subtokens no llevan prefijo ## sino que usan el carácter ▁ para
    indicar inicio de palabra. La función tokenizar_y_alinear de utils_ner
    maneja ambos casos correctamente mediante word_ids().
    """
    print(f"\n{'='*55}")
    print(f"  Entrenando XLM-RoBERTa  |  batch_size={batch_size}")
    print(f"{'='*55}")

    output_dir.mkdir(parents=True, exist_ok=True)

    modelo = AutoModelForTokenClassification.from_pretrained(
        MODELO_BASE,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=hf_repo,
        hub_strategy="end",
        fp16=True,       # precisión mixta: reduce memoria GPU ~30% sin perder calidad
        report_to="none",
    )

    trainer = Trainer(
        model=modelo,
        args=args,
        train_dataset=dataset_tokenizado["train"],
        eval_dataset=dataset_tokenizado["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    trainer.train()

    print(f"\n  Evaluando sobre conjunto de prueba...")
    metricas_test = trainer.evaluate(dataset_tokenizado["test"])
    print(f"  F1 test: {metricas_test.get('eval_f1', 0):.4f}")

    reporte_path = REPORTS_DIR / f"xlmroberta_bs{batch_size}_report.json"
    with open(reporte_path, "w") as f:
        json.dump(metricas_test, f, indent=2)
    print(f"  ✔ Reporte guardado: {reporte_path}")

    trainer.push_to_hub(commit_message=f"XLM-RoBERTa NER próstata — batch_size={batch_size}")
    print(f"  ✔ Modelo subido a Hugging Face: {hf_repo}")

    return metricas_test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    autenticar_hf()

    print("\n[1/4] Cargando dataset...")
    dataset = cargar_dataset(DATA_DIR)
    mostrar_estadisticas(dataset)

    print("\n[2/4] Cargando tokenizador XLM-RoBERTa...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    print("\n[3/4] Tokenizando y alineando etiquetas...")
    dataset_tok = dataset.map(
        lambda ejemplos: tokenizar_y_alinear(ejemplos, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=["tokens", "labels"],
    )

    print("\n[4/4] Entrenando con batch sizes:", BATCH_SIZES)
    resultados = {}

    for bs in BATCH_SIZES:
        output_dir = OUTPUT_BASE / f"bs{bs}"
        hf_repo    = f"{HF_REPO_BASE}-bs{bs}"
        metricas   = entrenar_con_batch(dataset_tok, tokenizer, bs, output_dir, hf_repo)
        resultados[f"bs{bs}"] = metricas

    # Resumen comparativo
    print("\n" + "=" * 55)
    print("  RESUMEN COMPARATIVO — XLM-RoBERTa NER PRÓSTATA")
    print("=" * 55)
    print(f"{'Batch size':<15} {'F1':>8} {'Precision':>12} {'Recall':>10}")
    print("-" * 55)
    for config, m in resultados.items():
        f1  = m.get("eval_f1", 0)
        pre = m.get("eval_precision", 0)
        rec = m.get("eval_recall", 0)
        print(f"{config:<15} {f1:>8.4f} {pre:>12.4f} {rec:>10.4f}")
    print("=" * 55)

    resumen_path = REPORTS_DIR / "xlmroberta_resumen.json"
    with open(resumen_path, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\n✔ Resumen guardado: {resumen_path}")


if __name__ == "__main__":
    main()
