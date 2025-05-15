# Clasificador de Sentimiento IMDB

Este proyecto permite entrenar y probar un modelo de análisis de sentimiento sobre reseñas de películas IMDB usando PyTorch y una interfaz gráfica sencilla.

## Requisitos
- Python 3.10+
- GPU NVIDIA (opcional, recomendado)
- Dependencias del proyecto (ver abajo)

## Instalación
1. Clona el repositorio y entra a la carpeta del proyecto.
2. Crea y activa un entorno virtual:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Instala todas las dependencias:
   ```powershell
   pip install -r requirements.txt
   ```
4. Descarga y descomprime el dataset IMDB en la carpeta `aclImdb/` (si no está incluida).

## Entrenamiento
Para entrenar el modelo desde cero o continuar el entrenamiento:
```powershell
python src/train.py
```
El modelo entrenado se guardará como `model.pt` en la raíz del proyecto.

## Evaluación
Para evaluar el modelo en el conjunto de test:
```powershell
python src/eval.py
```
Verás la precisión y pérdida final en consola.

## Interfaz gráfica
Para probar el modelo con tus propios textos usando una interfaz sencilla:
```powershell
python src/sentiment_gui.py
```
Se abrirá una ventana donde puedes escribir una reseña y ver la predicción.

