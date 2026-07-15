import os
from transformers import pipeline

# Ruta apuntando a tus archivos del modelo
MODEL_PATH = "./outputs/modelBetoSarcasmo/beto_bs16_best"

print("Cargando el modelo de detección de sarcasmo")

try:
    # Cargamos el clasificador
    classifier = pipeline(
        "text-classification", 
        model=MODEL_PATH, 
        tokenizer=MODEL_PATH
    )
    print("¡Modelo cargado con éxito!")
    print("================================================================")
    print("Escribe una frase para evaluar si contiene SARCASMO.")
    print("(Para salir del programa, escribe la palabra 'salir' o presiona Ctrl+C)")
    print("================================================================\n")
except Exception as e:
    print(f"Error al cargar el modelo. Detalle: {e}")
    exit()

# Bucle interactivo por consola
while True:
    # Captura lo que escribas en la terminal
    entrada_usuario = input("Tu oración > ")
    
    # Condición de salida limpia
    if entrada_usuario.strip().lower() == 'salir':
        print("\n¡Nos vemos! Cerrando el detector de sarcasmo.")
        break
    
    # Evita evaluar entradas vacías
    if not entrada_usuario.strip():
        continue
        
    # Realiza la predicción
    prediccion = classifier(entrada_usuario)[0]
    etiqueta = prediccion['label']
    confianza = prediccion['score'] * 100
    
    # Interpretamos la salida numérica de BETO
    resultado_texto = "🔴 SARCASMO DETECTADO" if etiqueta == "LABEL_1" else "🟢 NO SARCÁSTICO"
    
    # Imprime el resultado en consola
    print(f"Resultado: {resultado_texto} | Confianza: {confianza:.2f}%\n")