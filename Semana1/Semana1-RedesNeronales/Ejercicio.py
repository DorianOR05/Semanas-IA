from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Datos de entrenamiento (1 = Positivo, 0 = Negativo)
textos_entrenamiento = [
    "Excelente servicio, me encantó la comida",
    "El producto es genial, funciona a la perfección",
    "Me gustó mucho la atención del personal",
    "Una experiencia maravillosa, lo recomiendo",
    "Pésimo servicio, la comida estaba fría",
    "El producto no sirve, llegó roto y defectuoso",
    "No me gustó para nada la atención, muy lenta",
    "Una experiencia horrible, no vuelvo jamás"
]
etiquetas = [1, 1, 1, 1, 0, 0, 0, 0]

# 2. Vectorización: Convertir el texto a una matriz de números
vectorizador = CountVectorizer()
X_entrenamiento = vectorizador.fit_transform(textos_entrenamiento)

# 3. Entrenamiento: El modelo aprende los patrones de las palabras
modelo = MultinomialNB()
modelo.fit(X_entrenamiento, etiquetas)

# 4. Predicción: Clasificar textos nuevos que el modelo no conoce
nuevos_comentarios = [
    "La comida estuvo genial y la atención fue buena", 
    "Qué horror de producto, no sirve para nada"
]

# Los nuevos textos se deben transformar con el mismo vectorizador
X_nuevos = vectorizador.transform(nuevos_comentarios)
predicciones = modelo.predict(X_nuevos)

# Mostrar los resultados
for texto, prediccion in zip(nuevos_comentarios, predicciones):
    resultado = "Positivo" if prediccion == 1 else "Negativo"
    print(f"Texto: '{texto}' -> Clasificación: {resultado}")