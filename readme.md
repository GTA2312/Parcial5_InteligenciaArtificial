# 🌍 Clasificador de Escenas (Intel Image Classification)

Este repositorio contiene el código fuente, los modelos y la aplicación de despliegue para el **Parcial 5 de Inteligencia Artificial (Deep Learning)** de la Institución Universitaria Colegio Mayor del Cauca.

El proyecto consiste en una aplicación web interactiva que utiliza una Red Neuronal Convolucional (CNN) profunda para clasificar fotografías en seis tipos de escenas naturales y urbanas: **Buildings, Forest, Glacier, Mountain, Sea y Street**.

---

## 👥 Equipo de Desarrollo (Grupo 5)
* ANAYA VELASCO YEISON ALEJANDRO
* CHICAIZA PALECHOR ANDRES FELIPE
* COLLAZOS ORTIZ CAMILA ANDREA
* LOPEZ LEON SANTIAGO
* RUANO CAPOTE CRISTIAN SANTIAGO
* TROYANO ANDRADE GABRIEL STEBAN

---

## 🧠 Detalles Técnicos de la Inteligencia Artificial

Para este desafío, se justificó y diseñó la siguiente arquitectura:

* **Tipo de Red:** Red Neuronal Convolucional (CNN), elegida por su capacidad para preservar la estructura espacial de los píxeles y extraer patrones complejos (bordes, texturas) sin aplanar la imagen original.
* **Modelo Base:** `EfficientNetB0` (Preentrenado en ImageNet).
* **Estrategia:** **Transfer Learning**. Se utilizó el modelo base como extractor de características (*Feature Extraction*) y posteriormente se aplicó **Fine-Tuning** (ajuste fino) desbloqueando las últimas 20 capas con una tasa de aprendizaje reducida (`1e-5`) para especializar la red en nuestro dataset específico.
* **Prevención de Sobreajuste:** Se integraron técnicas de *Data Augmentation* (Rotación, Zoom, Volteo Horizontal, Contraste), *Dropout* (30%) y *Early Stopping* durante el entrenamiento.
* **Despliegue:** Streamlit (Aplicando el mismo preprocesamiento del entrenamiento).

---

## 📂 Estructura del Proyecto

```text
├── app.py                 # Código principal de la aplicación web (Streamlit)
├── app.ipynb              # Notebook de Jupyter con el entrenamiento de la CNN y métricas
├── requirements.txt       # Dependencias y versiones exactas de las librerías
├── .gitignore             # Archivos ignorados por Git (Datasets y Entornos Virtuales)
├── modelo/
│   ├── modelo.keras       # Modelo entrenado y exportado
│   └── clases.json        # Diccionario con el orden exacto de las clases
└── README.md              # Documentación del proyecto
```

*(Nota: Los datasets originales de entrenamiento y prueba han sido excluidos de este repositorio debido a su tamaño, utilizando el archivo `.gitignore`).*

---

## 🚀 Guía de Instalación y Despliegue Local

Sigue estos pasos para ejecutar la aplicación web en tu propio computador:

**1. Clonar el repositorio**
```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```
*(Cambia la URL por el enlace real de tu repositorio una vez publicado).*

**2. Crear y activar el entorno virtual**
Para Windows (Git Bash / PowerShell):
```bash
python -m venv venv
source venv/Scripts/activate
```

**3. Instalar las dependencias**
```bash
pip install -r requirements.txt
```

**4. Ejecutar la aplicación**
```bash
streamlit run app.py
```
El navegador se abrirá automáticamente en `http://localhost:8501`. 

---

## 📊 Dataset Utilizado
Los datos originales pertenecen al reto **Intel Image Classification** publicado originalmente por Analytics Vidhya. Contiene imágenes a color de 150x150 píxeles distribuidas en 6 categorías.