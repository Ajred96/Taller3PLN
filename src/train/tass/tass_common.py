"""
Utilidades comunes para el finetuning de clasificación TASS (3 clases: N, NEU, P).

Reutilizado por los runners:
    - train_beto.py
    - train_xlnet.py
    - train_xlmroberta.py

El dataset procesado (data/processed/tass/) solo trae train.json y test.json,
por lo que aquí se genera el conjunto de validación (15% estratificado del train)
para poder aplicar Early Stopping tal como pide el taller.
"""
import json
from pathlib import Path

import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ==========================================
# 0. CONSTANTES DEL PROBLEMA
# ==========================================
ROOT_DIR = Path(__file__).resolve().parents[3]
TASS_DIR = ROOT_DIR / "data" / "processed" / "tass"
MODELS_DIR = ROOT_DIR / "outputs" / "models" / "tass"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# Orden fijo de etiquetas -> id (consistente entre todos los modelos)
LABELS = ["N", "NEU", "P"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

SEED = 42
VAL_SIZE = 0.15


# ==========================================
# 1. CARGA + SPLIT ESTRATIFICADO
# ==========================================
def load_datasets():
    """Devuelve un DatasetDict con train/validation/test y la columna 'label_id' (int)."""
    def _read(name):
        with open(TASS_DIR / name, encoding="utf-8") as fh:
            rows = json.load(fh)
        return Dataset.from_list(rows)

    train_full = _read("train.json")
    test = _read("test.json")

    # Mapear etiqueta string -> id entero
    def add_label_id(ex):
        return {"label_id": LABEL2ID[ex["label"]]}

    train_full = train_full.map(add_label_id)
    test = test.map(add_label_id)

    # Split estratificado 85/15 para crear validación (Early Stopping)
    splits = train_full.train_test_split(
        test_size=VAL_SIZE, seed=SEED, stratify_by_column="label_id"
    )

    return DatasetDict(
        train=splits["train"],
        validation=splits["test"],
        test=test,
    )


# ==========================================
# 2. TOKENIZACIÓN
# ==========================================
def tokenize(datasets, tokenizer, max_length=128):
    def _tok(examples):
        result = tokenizer(examples["text"], truncation=True, max_length=max_length)
        result["labels"] = examples["label_id"]
        return result

    return datasets.map(
        _tok,
        batched=True,
        remove_columns=["text", "label", "label_id"],
    )


# ==========================================
# 3. MÉTRICAS (3 clases -> macro / weighted / accuracy)
# ==========================================
_accuracy = evaluate.load("accuracy")
_f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    labels = labels.astype(int)
    return {
        "accuracy": _accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": _f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "f1_weighted": _f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }


# ==========================================
# 4. RUNNER GENÉRICO (recorre batch sizes)
# ==========================================
def run_experiment(
    model_checkpoint,
    tag,
    batch_sizes=(8, 16, 32),
    epochs=10,
    learning_rate=1e-5,
    max_length=128,
    freeze_backbone_attr=None,
    unfreeze_last=6,
    grad_accum=1,
    per_device_cap=None,
    fp16=False,
    run=False,
):
    """
    Entrena `model_checkpoint` sobre TASS para cada batch size y escribe un
    reporte comparativo en outputs/reports/tass_<tag>_report.txt.

    Parámetros pensados para modelos grandes (XLM-RoBERTa-large):
      - freeze_backbone_attr: nombre del submódulo base a congelar (ej. "roberta").
      - unfreeze_last: nº de capas finales del encoder a descongelar.
      - grad_accum: acumulación de gradiente para simular batch efectivo.
      - per_device_cap: batch físico máximo por GPU (VRAM). Si se define, el batch
        efectivo se logra con gradient_accumulation_steps.
      - fp16: precisión mixta para acelerar en GPU NVIDIA.
      - run: si False solo estructura los scripts (útil para sustentación sin GPU).
    """
    print(f"\n########## TASS · {tag} ({model_checkpoint}) ##########")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    datasets = tokenize(load_datasets(), tokenizer, max_length=max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    resultados = []

    for bs in batch_sizes:
        print(f"\n=== {tag} · Batch Size = {bs} ===")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        # Congelamiento opcional para modelos grandes
        if freeze_backbone_attr and hasattr(model, freeze_backbone_attr):
            backbone = getattr(model, freeze_backbone_attr)
            for p in backbone.parameters():
                p.requires_grad = False
            layers = backbone.encoder.layer
            for i in range(len(layers) - unfreeze_last, len(layers)):
                for p in layers[i].parameters():
                    p.requires_grad = True
            if getattr(backbone, "pooler", None) is not None:
                for p in backbone.pooler.parameters():
                    p.requires_grad = True
            print(f"-> Backbone '{freeze_backbone_attr}' congelado; "
                  f"descongeladas últimas {unfreeze_last}/{len(layers)} capas")

        # Batch físico vs efectivo (control de VRAM)
        if per_device_cap and bs > per_device_cap:
            per_device_bs = per_device_cap
            accum = max(grad_accum, bs // per_device_cap)
        else:
            per_device_bs = bs
            accum = grad_accum

        out_dir = MODELS_DIR / f"{tag}_bs{bs}"
        training_args = TrainingArguments(
            output_dir=str(out_dir / "results"),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=per_device_bs,
            gradient_accumulation_steps=accum,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=fp16,
            logging_steps=100,
            save_total_limit=1,
            seed=SEED,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        if not run:
            print(f"[dry-run] {tag} bs{bs} estructurado (efectivo={per_device_bs}x{accum}). "
                  f"Poner run=True para entrenar.")
            continue

        trainer.train()

        # Evaluación final sobre TEST (métricas que pide el taller)
        test_metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
        best = trainer.model
        best.save_pretrained(out_dir / "best")
        tokenizer.save_pretrained(out_dir / "best")

        resultados.append({
            "batch_size": bs,
            "test_accuracy": test_metrics.get("test_accuracy"),
            "test_f1_macro": test_metrics.get("test_f1_macro"),
            "test_f1_weighted": test_metrics.get("test_f1_weighted"),
        })
        print(f"-> {tag} bs{bs} TEST: {test_metrics}")

    _write_report(tag, model_checkpoint, resultados)
    return resultados


def _write_report(tag, checkpoint, resultados):
    report_path = REPORTS_DIR / f"tass_{tag}_report.txt"
    lines = [
        f"REPORTE TASS · {tag} ({checkpoint})",
        "Clases: N / NEU / P  |  Métrica de selección: F1 macro  |  Early Stopping paciencia=5",
        "Baseline BETO a superar: 64% f1",
        "-" * 72,
        f"{'BatchSize':<12}{'Accuracy':<12}{'F1 Macro':<12}{'F1 Weighted':<12}",
    ]
    for r in resultados:
        lines.append(
            f"{r['batch_size']:<12}"
            f"{_fmt(r['test_accuracy']):<12}"
            f"{_fmt(r['test_f1_macro']):<12}"
            f"{_fmt(r['test_f1_weighted']):<12}"
        )
    if not resultados:
        lines.append("(dry-run: sin métricas; ejecutar con run=True en GPU)")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReporte escrito en: {report_path}")


def _fmt(v):
    return f"{v*100:.2f}%" if isinstance(v, (int, float)) else "-"
