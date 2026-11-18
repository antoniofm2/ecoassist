"""Inferência do classificador visual (MobileNetV2 ajustado no TrashNet)."""
from pathlib import Path

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
PESOS = Path(__file__).resolve().parents[2] / "resultados" / "mobilenetv2_trashnet.pt"


def _transform():
    from torchvision import transforms as T
    return T.Compose([T.Resize((224, 224)), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def construir_modelo(pretreinado: bool = True):
    import torch.nn as nn
    from torchvision import models
    w = models.MobileNet_V2_Weights.DEFAULT if pretreinado else None
    m = models.mobilenet_v2(weights=w)
    m.classifier[1] = nn.Linear(m.last_channel, len(CLASSES))
    return m


class ClassificadorVisual:
    def __init__(self, pesos: Path = PESOS):
        import torch
        self.torch = torch
        self.modelo = construir_modelo(pretreinado=False)
        self.modelo.load_state_dict(torch.load(pesos, map_location="cpu"))
        self.modelo.eval()
        self.tf = _transform()

    def prever(self, imagem, k: int = 3):
        """imagem: PIL.Image. Retorna lista [(classe, prob)] ordenada, top-k."""
        x = self.tf(imagem.convert("RGB")).unsqueeze(0)
        with self.torch.no_grad():
            p = self.torch.softmax(self.modelo(x), dim=1)[0]
        top = p.argsort(descending=True)[:k]
        return [(CLASSES[i], float(p[i])) for i in top]
