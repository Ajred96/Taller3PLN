import os
from transformers import pipeline, AutoTokenizer

# 1. Definimos la ruta relativa exacta (Corregido: XLM en lugar de XML)
RUTA_RELATIVA = "./outputs/modelBetoSarcasmo/XLM-RoBERTa_bs16_best"

# 2. Convertimos a ruta absoluta normalizada de Windows
# Hugging Face acepta perfectamente rutas absolutas nativas (ej: "D:/Univalle/...")
MODEL_PATH = os.path.abspath(RUTA_RELATIVA).replace(os.sep, "/")

print("Cargando el modelo XLM-RoBERTa de detección de sarcasmo...")

try:
    # Cargamos el tokenizador oficial
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    
    # Cargamos el clasificador apuntando a la ruta absoluta corregida
    classifier = pipeline(
        "text-classification", 
        model=MODEL_PATH, 
        tokenizer=tokenizer
    )
    print("¡Modelo cargado con éxito!")
    print("================================================================")
    print("Escribe una frase para evaluar si contiene SARCASMO (XLM-RoBERTa BS = 16).")
    print("(Para salir del programa, escribe la palabra 'salir' o presiona Ctrl+C)")
    print("================================================================\n")
except Exception as e:
    print(f"Error al cargar el modelo. Verifica que la carpeta '{RUTA_RELATIVA}' contenga los archivos config.json y model.safetensors sueltos. Detalle: {e}")
    exit()

# Bucle interactivo por consola
while True:
    entrada_usuario = input("Tu oración > ")
    
    if entrada_usuario.strip().lower() == 'salir':
        print("\n¡Nos vemos! Cerrando el detector de sarcasmo.")
        break
    
    if not entrada_usuario.strip():
        continue
        
    prediccion = classifier(entrada_usuario)[0]
    print(f"DEBUG - Predicción cruda: {prediccion}")
    etiqueta = prediccion['label']
    confianza = prediccion['score'] * 100
    
    
    resultado_texto = "🔴 SARCASMO DETECTADO" if etiqueta == "LABEL_1" else "🟢 NO SARCÁSTICO"
    
    print(f"Resultado: {resultado_texto} | Confianza: {confianza:.2f}%\n")