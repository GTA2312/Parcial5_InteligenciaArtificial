import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

# Definimos el tamaño exacto con el que entrenamos el modelo[cite: 9, 12]
TAMANO = (150, 150)

st.set_page_config(page_title="Clasificador de Escenas", page_icon="🌍")

# @st.cache_resource evita que el modelo se recargue cada vez que se presiona un botón
@st.cache_resource
def cargar_modelo():
    # Cargamos la arquitectura y los pesos[cite: 4]
    modelo = keras.models.load_model("modelo/modelo.keras")
    # Cargamos el orden exacto de las clases[cite: 4]
    with open("modelo/clases.json", encoding="utf-8") as f:
        clases = json.load(f)
    return modelo, clases

def preprocesar(imagen):
    # Aseguramos que la imagen tenga 3 canales (RGB) y el tamaño correcto[cite: 5]
    imagen = imagen.convert("RGB").resize(TAMANO)
    arreglo = np.array(imagen, dtype="float32")
    
    # IMPORTANTE: No dividimos por 255.0 porque EfficientNetB0 (que usamos de base) 
    # ya incluye su propia capa de normalización de 0 a 255[cite: 1, 5].
    # Agregamos una dimensión para simular un "lote" (batch) de 1 imagen: (1, 150, 150, 3)[cite: 5]
    return np.expand_dims(arreglo, axis=0)

st.title("🌍 Clasificador de Escenas (Intel Image)")
st.write("Sube una fotografía de la carpeta `seg_pred` o tuya propia, y el modelo predecirá el tipo de escena.")

# Cargamos el modelo y las clases en memoria
modelo, clases = cargar_modelo()

# Creamos el botón para subir archivos[cite: 5]
archivo = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    # Mostramos la imagen en pantalla[cite: 5]
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen a clasificar", width=400)
    
    # Botón para ejecutar la predicción[cite: 5]
    if st.button("Predecir Escena"):
        # Preprocesamos y pasamos la imagen al modelo
        salida = modelo.predict(preprocesar(imagen), verbose=0)[0]
        
        # Obtenemos el índice del valor más alto en las probabilidades[cite: 5]
        indice = int(np.argmax(salida))
        
        # Mostramos la clase ganadora y su nivel de confianza[cite: 5, 9]
        st.success(f"Predicción Principal: **{clases[indice]}**")
        st.metric("Confianza de la red", f"{salida[indice] * 100:.2f} %")
        
        # Mostramos el top 3 de probabilidades como valor agregado[cite: 5]
        st.subheader("Top 3 de clases más probables:")
        for i in np.argsort(salida)[::-1][:3]:
            st.write(f"- {clases[i]}: {salida[i] * 100:.2f} %")
            st.progress(float(salida[i]))