"""
Finetuning de BETO (bert-base-spanish-wwm-cased) sobre TASS.
Recorre batch sizes 8, 16, 32 con Early Stopping (paciencia=5).

Uso:
    python src/train/tass/train_beto.py          # dry-run (estructura, sin GPU)
    RUN=1 python src/train/tass/train_beto.py    # entrena de verdad
"""
import os
from tass_common import run_experiment

if __name__ == "__main__":
    run_experiment(
        model_checkpoint="dccuchile/bert-base-spanish-wwm-cased",
        tag="beto",
        batch_sizes=(8, 16, 32),
        epochs=10,
        learning_rate=1e-5,
        run=os.environ.get("RUN") == "1",
    )
