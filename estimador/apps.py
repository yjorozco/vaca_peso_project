import os
import sys
import torch
from django.apps import AppConfig
from django.conf import settings

# Variable de entorno para confiar en UniDepth sin pedir (y/N)
os.environ['TORCH_HUB_TRUSTED_REPOS'] = 'lpiccinelli-eth/UniDepth'

class EstimadorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'estimador'

    device = None
    detectron_predictor = None
    unidepth_model = None
    detector_postura = None
    regresor_a = None
    regresor_b = None

    def ready(self):
        # NO cargar modelos pesados si solo estamos ejecutando migraciones o comandos de gestión
        comandos_migracion = ['makemigrations', 'migrate', 'collectstatic', 'createsuperuser']
        if any(cmd in sys.argv for cmd in comandos_migracion):
            return

        # Evitar doble carga en el reloader de desarrollo
        if os.environ.get('RUN_MAIN') != 'true' and 'runserver' in sys.argv:
            return

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"--> [PyTorch] Inicializando modelos en: {self.device}")

        # 1. Detectron2
        try:
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2 import model_zoo

            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.detectron_predictor = DefaultPredictor(cfg)
            print("--> [Detectron2] Cargado exitosamente.")
        except Exception as e:
            print(f"Advertencia: Detectron2 no cargado ({e})")

        # 2. UniDepth
        try:
            self.unidepth_model = torch.hub.load(
                "lpiccinelli-eth/UniDepth",
                "UniDepth",
                version="v2",
                backbone="vitl14",
                pretrained=True,
                trust_repo=True  # Evita el prompt interactivo y/N
            )
            self.unidepth_model = self.unidepth_model.to(self.device).eval()
            if hasattr(self.unidepth_model, "resolution_level"):
                self.unidepth_model.resolution_level = 0
            print("--> [UniDepth] Cargado exitosamente.")
        except Exception as e:
            print(f"Advertencia: UniDepth no cargado ({e})")

        # 3. Modelos .pth propios
        from .ml_models.architectures import CowPostureClassifier, CowWeightRegressorFiLM

        weights_dir = os.path.join(settings.BASE_DIR, 'estimador', 'ml_models')
        path_postura = os.path.join(weights_dir, 'best_cow_posture_model_v1.6_RESNET50.pth')
        path_reg_0 = os.path.join(weights_dir, 'best_cow_weight_model_0_film_v21.pth')
        path_reg_3 = os.path.join(weights_dir, 'best_cow_weight_model_3_film_v21.pth')

        self.detector_postura = CowPostureClassifier(num_classes=4).to(self.device)
        if os.path.exists(path_postura):
            self.detector_postura.load_state_dict(torch.load(path_postura, map_location=self.device))
        self.detector_postura.eval()

        self.regresor_a = CowWeightRegressorFiLM().to(self.device)
        if os.path.exists(path_reg_0):
            self.regresor_a.load_state_dict(torch.load(path_reg_0, map_location=self.device))
        self.regresor_a.eval()

        self.regresor_b = CowWeightRegressorFiLM().to(self.device)
        if os.path.exists(path_reg_3):
            self.regresor_b.load_state_dict(torch.load(path_reg_3, map_location=self.device))
        self.regresor_b.eval()

        print("--> [PyTorch] Todos los modelos listos para inferencia.")
