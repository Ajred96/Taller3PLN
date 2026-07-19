"""
Demo TASS: clasifica una lista de frases de ejemplo y muestra la prediccion.
Ideal para grabar el video (salida limpia, sin escribir en vivo).

Uso:
    python src/predict/tass/demo_tass.py            # modelo ganador (XLM-R)
    python src/predict/tass/demo_tass.py --tres     # compara los 3 modelos
"""
import sys
from transformers import pipeline

GANADOR = "gustavoa6791/xlmroberta-tass-sentiment-univalle-pln"
MODELOS = {
    "BETO":  "gustavoa6791/beto-tass-sentiment-univalle-pln",
    "XLNet": "gustavoa6791/xlnet-tass-sentiment-univalle-pln",
    "XLM-R": "gustavoa6791/xlmroberta-tass-sentiment-univalle-pln",
}

EMOJI = {"N": "🔴 Negativo", "NEU": "🟡 Neutro", "P": "🟢 Positivo"}

EJEMPLOS = [
    "Qué día tan maravilloso, todo me salió perfecto",
    "Odio cuando el bus se retrasa y llego tarde",
    "Mañana hay reunión de trabajo a las 10am",
    "No sé si me gusta o no, es raro",
    "Me encantó la película, la recomiendo muchísimo",
    "Estoy harto de esta situación, ya no aguanto más",
    "El informe se entrega el viernes según lo acordado",
    "Qué alegría verte, te extrañaba un montón",
]


def clasificar(clf, texto):
    r = clf(texto)[0]
    return r["label"], r["score"] * 100


def main():
    tres = "--tres" in sys.argv

    if tres:
        clfs = {n: pipeline("text-classification", model=r, tokenizer=r)
                for n, r in MODELOS.items()}
        print("\n" + "=" * 90)
        print(f"{'Frase':<52}" + "".join(f"{n:<13}" for n in MODELOS))
        print("=" * 90)
        for txt in EJEMPLOS:
            fila = f"{txt[:50]:<52}"
            for n in MODELOS:
                lab, conf = clasificar(clfs[n], txt)
                fila += f"{lab} ({conf:.0f}%)".ljust(13)
            print(fila)
        print("=" * 90)
    else:
        print(f"\nModelo: {GANADOR} (ganador)\n")
        clf = pipeline("text-classification", model=GANADOR, tokenizer=GANADOR)
        print("=" * 70)
        for txt in EJEMPLOS:
            lab, conf = clasificar(clf, txt)
            print(f"  \"{txt}\"")
            print(f"    -> {EMOJI.get(lab, lab)}   (confianza {conf:.1f}%)\n")
        print("=" * 70)


if __name__ == "__main__":
    main()
