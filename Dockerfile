FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HUB_TRUSTED_REPOS=lpiccinelli-eth/UniDepth

WORKDIR /app

# Instalar dependencias del sistema y herramientas de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*



RUN mkdir -p /root/.torch/iopath_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/ && \
    curl -k -L "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" \
    -o /root/.torch/iopath_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
# setuptools<70 asegura la disponibilidad de pkg_resources para Detectron2
RUN pip install --no-cache-dir --upgrade pip "setuptools<70" wheel

# Instalar PyTorch CPU
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar Detectron2 y UniDepth
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
RUN pip install --no-cache-dir "git+https://github.com/lpiccinelli-eth/UniDepth.git"

# Instalar resto de requerimientos
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN mkdir -p /app/media/uploads /app/static
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]