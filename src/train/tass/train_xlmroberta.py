"""
Finetuning de XLM-RoBERTa-large (~560M params) sobre TASS.
Recorre batch sizes efectivos 8, 16, 32 con Early Stopping (paciencia=5).

Por restricciones de VRAM se congela el cuerpo y se descongelan las últimas 6
capas del encoder (mismo enfoque que el experimento de Sarcasmo). El batch físico
se limita con per_device_cap=4 y el batch efectivo se alcanza vía gradient
accumulation. fp16 activado para GPU NVIDIA / Colab.

Uso:
    python src/train/tass/train_xlmroberta.py
    RUN=1 python src/train/tass/train_xlmroberta.py
"""
import os
from tass_common import run_experiment

if __name__ == "__main__":
    run_experiment(
        model_checkpoint="xlm-roberta-large",
        tag="xlmroberta",
        batch_sizes=(8, 16, 32),
        epochs=8,
        learning_rate=1e-5,
        freeze_backbone_attr="roberta",
        unfreeze_last=6,
        per_device_cap=4,
        fp16=True,
        run=os.environ.get("RUN") == "1",
    )
