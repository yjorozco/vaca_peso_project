# BovinoWeight AI - Sistema Monocular de Fenotipado y Estimación de Peso Bovino

Aplicación monolítica MVC construida en **Django**, **SQLite**, **HTML5/CSS3** (con estilo UI inspirado en Cattlytics) e inferencia profunda en **PyTorch**.

## Estructura de archivos:
- `estimador/ml_models/architectures.py`: Arquitecturas de red (FiLMLayer, CowWeightRegressorFiLM, CowPostureClassifier).
- `estimador/ml_models/*.pth`: Colocar aquí los 3 archivos de pesos entrenados:
  - `best_cow_weight_model_0_film_v21.pth`
  - `best_cow_weight_model_3_film_v21.pth`
  - `best_cow_posture_model_v1.6_RESNET50.pth`
- `estimador/apps.py`: Carga única de modelos en RAM/GPU al arrancar.
- `estimador/views.py`: Pipeline Detectron2 -> UniDepth -> Bitwise AND -> Postura -> FiLM Regressor.

## Puesta en marcha:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```
