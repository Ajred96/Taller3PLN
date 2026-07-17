import os
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback
)

# ==========================================
# 0. CONFIGURACIÓN DE RUTAS LOCALES
# ==========================================

PATH_TRAIN = "./Sarcasmo/train.json"
PATH_VAL = "./Sarcasmo/validation.json"
DIR_RESULTADOS = "./resultados_xlmr_sarcasmo"
DIR_MODELO_FINAL = "./modelo_final_xlmr"

def main():
    print("=== 1. Cargando el Dataset ===")
    data_files = {
        "train": PATH_TRAIN,
        "validation": PATH_VAL
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    print(raw_datasets)

    print("\n=== 2. Inicializando Tokenizador ===")
    model_checkpoint = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )

    print("Tokenizando textos...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Convertimos las etiquetas a enteros
    tokenized_datasets = tokenized_datasets.map(lambda x: {"label": int(x["label"])})

    print("--- Validación de formato de datos ---")
    print("Columnas finales:", tokenized_datasets["train"].column_names)
    print("Tipo de dato de label (Debe ser int):", type(tokenized_datasets["train"][0]["label"]))

    print("\n=== 3. Configurando Métricas y Data Collator ===")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]
        return {
            "accuracy": acc,
            "f1": f1
        }

    print("\n=== 4. Cargando Modelo y Configurando Capas ===")
    # Cargamos el modelo base
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2
    )

    # Paso A: Congelamos absolutamente todo el cuerpo de RoBERTa
    for param in model.roberta.parameters():
        param.requires_grad = False

    # Paso B: Descongelamos únicamente las últimas 6 capas del encoder
    capas_a_descongelar = 6
    total_capas = len(model.roberta.encoder.layer) # 24

    for i in range(total_capas - capas_a_descongelar, total_capas):
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = True

    # Paso C: Descongelamos el pooler final si existe
    if hasattr(model.roberta, 'pooler') and model.roberta.pooler is not None:
        for param in model.roberta.pooler.parameters():
            param.requires_grad = True

    print(f"-> Modelo cargado con éxito.")
    print(f"-> Capas del encoder descongeladas para entrenamiento: {capas_a_descongelar} de {total_capas}")

    print("\n=== 5. Configurando Argumentos de Entrenamiento ===")
    training_args = TrainingArguments(
        output_dir=DIR_RESULTADOS,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4, # Batch size físico de 4 para cuidar tu VRAM local
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2, # Simula un Batch size efectivo de 8
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,                     # Activado para acelerar el cálculo en GPUs NVIDIA locales
        logging_steps=50,
        save_total_limit=1,            # Mantiene solo el mejor checkpoint para no llenar tu disco
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("\n=== 6. ¡Iniciando el entrenamiento de XLM-RoBERTa-large! ===")
    trainer.train()

    print("\n=== 7. Guardando el modelo final optimizado ===")
    # Guardamos los pesos y el tokenizador limpios en la ruta final
    model.save_pretrained(DIR_MODELO_FINAL)
    tokenizer.save_pretrained(DIR_MODELO_FINAL)
    print(f"¡Proceso completado! Modelo final listo en: '{DIR_MODELO_FINAL}'")

if __name__ == "__main__":
    main()