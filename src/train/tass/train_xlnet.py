"""
Finetuning de XLNet sobre TASS. Recorre batch sizes 8, 16, 32 con Early Stopping.

Nota: no existe un XLNet monolingüe español ampliamente distribuido. Se usa el
checkpoint base 'xlnet-base-cased'; su tokenizador SentencePiece maneja español
razonablemente. Documentar en el análisis (Punto 5) el efecto de no ser nativo.

Uso:
    python src/train/tass/train_xlnet.py
    RUN=1 python src/train/tass/train_xlnet.py
"""
import os
from tass_common import run_experiment

if __name__ == "__main__":
    run_experiment(
        model_checkpoint="xlnet-base-cased",
        tag="xlnet",
        batch_sizes=(8, 16, 32),
        epochs=10,
        learning_rate=2e-5,
        run=os.environ.get("RUN") == "1",
    )
