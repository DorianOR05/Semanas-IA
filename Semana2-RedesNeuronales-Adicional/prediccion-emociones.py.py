import os
import numpy as np
import cv2
import json
import h5py

import tensorflow as tf
from tensorflow.keras import layers, Sequential


ruta = os.path.dirname(os.path.abspath(__file__))

cascada_caras = cv2.CascadeClassifier(os.path.join(ruta, 'haarcascade_frontalface_default.xml'))

ruta_h5 = os.path.join(ruta, 'reconcedor-facial.h5')

with h5py.File(ruta_h5, 'r') as f:
    raw = f.attrs['model_config']
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    config = json.loads(raw)
    nombres_capas = [capa['config']['name'] for capa in config['config']['layers']]

modelo = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 3), name=nombres_capas[0]),
    layers.MaxPooling2D((2, 2), name=nombres_capas[1]),
    layers.Conv2D(64, (3, 3), activation='relu', name=nombres_capas[2]),
    layers.MaxPooling2D((2, 2), name=nombres_capas[3]),
    layers.Conv2D(128, (3, 3), activation='relu', name=nombres_capas[4]),
    layers.MaxPooling2D((2, 2), name=nombres_capas[5]),
    layers.Conv2D(128, (3, 3), activation='relu', name=nombres_capas[6]),
    layers.MaxPooling2D((2, 2), name=nombres_capas[7]),
    layers.Flatten(name=nombres_capas[8]),
    layers.Dropout(0.5, name=nombres_capas[9]),
    layers.Dense(512, activation='relu', name=nombres_capas[10]),
    layers.Dense(7, activation='softmax', name=nombres_capas[11])
])

modelo.load_weights(ruta_h5, by_name=True)

etiquetas = ['ira', 'contento', 'disgusto', 'miedo', 'feliz', 'tristeza', 'sorpresa']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    caras = cascada_caras.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in caras:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

        rostro = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        rostro = cv2.resize(rostro, (48, 48))
        rostro = np.stack([rostro] * 3, axis=-1)
        rostro = np.expand_dims(rostro, axis=0)

        clases = modelo.predict(rostro, verbose=0)[0]
        indice = np.argmax(clases)
        confianza = clases[indice]
        texto = f'{etiquetas[indice]} ({confianza:.2f})'
        print(clases.round(3), '->', texto)

        cv2.putText(gray, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
