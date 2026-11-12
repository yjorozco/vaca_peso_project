import os
import base64
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.apps import apps
import ssl

# 👉 PONLO AQUÍ (al principio de todo)
ssl._create_default_https_context = ssl._create_unverified_context
from .forms import ImagenVacaForm

transform_normalizacion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def procesar_pipeline_peso(uploaded_file):
    """
    Procesa la imagen directamente en memoria sin escribir en disco ni BD.
    """
    config = apps.get_app_config('estimador')
    device = config.device

    # 1. Cargar imagen directamente desde el buffer en memoria
    imagen_pil = Image.open(uploaded_file).convert("RGB")
    imagen_pil_resized = imagen_pil.resize((224, 224), Image.Resampling.BILINEAR)
    imagen_np_uint8 = np.array(imagen_pil_resized)

    # 2. Segmentación con Detectron2
    mask_binaria = np.ones((224, 224), dtype=bool)
    if config.detectron_predictor is not None:
        try:
            outputs = config.detectron_predictor(imagen_np_uint8)
            instances = outputs["instances"].to("cpu")
            pred_masks = instances.pred_masks.numpy()
            pred_classes = instances.pred_classes.numpy()
            pred_scores = instances.scores.numpy()

            cow_indices = np.where(pred_classes == 19)[0]
            if len(cow_indices) > 0:
                best_idx = cow_indices[np.argmax(pred_scores[cow_indices])]
                mask_binaria = pred_masks[best_idx]
        except Exception as e:
            print(f"Error en Detectron2: {e}")

    # 3. Profundidad métrica con UniDepth
    distancia_medida = 3.0
    if config.unidepth_model is not None:
        try:
            with torch.no_grad():
                rgb_tensor_unidepth = transforms.ToTensor()(imagen_pil).unsqueeze(0).to(device)
                predictions = config.unidepth_model.infer(rgb_tensor_unidepth)
                depth_map_native = predictions["depth"].squeeze().cpu().numpy()

            depth_map = cv2.resize(depth_map_native, (224, 224), interpolation=cv2.INTER_LINEAR)
            mask_indices = mask_binaria > 0
            if np.any(mask_indices):
                distancia_medida = float(np.median(depth_map[mask_indices]))
            else:
                distancia_medida = float(np.median(depth_map))
        except Exception as e:
            print(f"Error en UniDepth: {e}")

    # 4. Enmascaramiento Bitwise AND
    image_original_pil_resized_lanczos = imagen_pil.resize((224, 224), Image.LANCZOS).convert('RGB')
    image_np_rgb_lanczos = np.array(image_original_pil_resized_lanczos)
    mask_binary_3ch_cv = cv2.cvtColor(mask_binaria.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    masked_color_np = cv2.bitwise_and(image_np_rgb_lanczos, mask_binary_3ch_cv)
    imagen_final_pil = Image.fromarray(masked_color_np)

    x_img = transform_normalizacion(imagen_final_pil).unsqueeze(0).to(device)

    # 5. Inferencia Postura y Regresión FiLM
    peso_final_kg = None
    nombre_postura = "Postura 0"
    obs = "Procesado"

    if config.detector_postura is not None:
        with torch.no_grad():
            pred_postura = config.detector_postura(x_img)
            clase_postura = torch.argmax(pred_postura, dim=1).item()

            if clase_postura == 0:
                nombre_postura = "Postura 0"
                media_train, std_train = 3.4006968412734224, 1.0542972617476725
                dist_std = (distancia_medida - media_train) / (std_train + 1e-8)
                dist_tensor = torch.tensor([[dist_std]], dtype=torch.float32).to(device)
                if config.regresor_a is not None:
                    peso_estimado = config.regresor_a(x_img, dist_tensor)
                    peso_final_kg = round(float(peso_estimado.item()), 2)
                    obs = "Estimado exitosamente con regresor Postura 0"
            elif clase_postura == 3:
                nombre_postura = "Postura 3"
                media_train, std_train = 4.251978052116183, 1.3484686672778392
                dist_std = (distancia_medida - media_train) / (std_train + 1e-8)
                dist_tensor = torch.tensor([[dist_std]], dtype=torch.float32).to(device)
                if config.regresor_b is not None:
                    peso_estimado = config.regresor_b(x_img, dist_tensor)
                    peso_final_kg = round(float(peso_estimado.item()), 2)
                    obs = "Estimado exitosamente con regresor Postura 3"
            else:
                nombre_postura = f"Otra ({clase_postura})"
                obs = "Advertencia: Postura no apta para estimación de peso"

    return {
        "postura": nombre_postura,
        "distancia": round(distancia_medida, 3),
        "peso_kg": peso_final_kg,
        "observaciones": obs
    }

def login_view(request):
    if request.user.is_authenticated:
        return redirect('estimar')
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('estimar')
    else:
        form = AuthenticationForm()
    return render(request, 'estimador/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def estimar_view(request):
    resultado = None
    if request.method == 'POST':
        form = ImagenVacaForm(request.POST, request.FILES)
        if form.is_valid():
            archivo_imagen = request.FILES['imagen']
            
            # Ejecutar pipeline directo desde memoria
            res = procesar_pipeline_peso(archivo_imagen)

            # Generar preview en Base64 para mostrarlo sin almacenar archivo
            archivo_imagen.seek(0)
            img_base64 = base64.b64encode(archivo_imagen.read()).decode('utf-8')
            img_data_uri = f"data:{archivo_imagen.content_type};base64,{img_base64}"

            resultado = {
                "postura": res["postura"],
                "distancia": res["distancia"],
                "peso_kg": res["peso_kg"],
                "obs": res["observaciones"],
                "imagen_url": img_data_uri
            }
    else:
        form = ImagenVacaForm()

    return render(request, 'estimador/estimar.html', {'form': form, 'resultado': resultado})