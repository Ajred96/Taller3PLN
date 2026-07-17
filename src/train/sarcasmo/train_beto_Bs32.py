import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback
)

# ==========================================
# 1. CARGA DE DATASETS
# ==========================================
# Usamos las rutas relativas de tu proyecto local para que sea transportable
data_files = {
    "train": "./data/processed/sarcasmo/train.json",
    "validation": "./data/processed/sarcasmo/validation.json"
}

print("Cargando el dataset...")
raw_datasets = load_dataset("json", data_files=data_files)
print(raw_datasets)

# ==========================================
# 2. TOKENIZACIÓN Y PADDING
# ==========================================
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    result = tokenizer(examples["text"], truncation=True)
    result["labels"] = [int(l) for l in examples["label"]]
    return result

print("Tokenizando los textos...")
tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=["text", "label"]  
)
print(tokenized_datasets)

# Inicializamos el colador de datos encargado del padding dinámico
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Padding configurado y listo para aplicarse dinámicamente.")

# ==========================================
# 3. DEFINICIÓN DE MÉTRICAS
# ==========================================
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    labels = labels.astype(int)

    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]

    return {
        "accuracy": acc,
        "f1": f1
    }

# ==========================================
# 4. CONFIGURACIÓN DEL ENTRENAMIENTO
# ==========================================
# 1. Cargamos el modelo 
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 2. Argumentos del Experimento 3 (Batch Size = 32)
training_args = TrainingArguments(
    output_dir="./outputs/modelBetoSarcasmo/results_beto_bs32", 
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,  
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    logging_steps=100,
    seed=42
)

# 3. Inicializamos el Trainer con la paciencia en 5 para este lote
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,          
    callbacks=[EarlyStoppingCallback(early_stopping_pvariance=5)] 
)

# ==========================================
# 5. EJECUCIÓN 
# ==========================================
if __name__ == "__main__":
    print("\n=== Iniciando el entrenamiento de BETO (Batch Size = 32) ===")
    # trainer.train() 
    print("Script estructurado correctamente para la sustentación.")