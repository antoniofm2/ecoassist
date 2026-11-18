"""Treino e avaliação: baseline HOG+SVM e MobileNetV2 (transfer learning).

Uso: python -m src.machine_learning.treinar_imagens --dados dados/trashnet --epocas 8
Estrutura esperada: dados/trashnet/<classe>/*.jpg (TrashNet, 6 classes).
Gera resultados/ (pesos, métricas.json, matrizes de confusão, curva de treino).
"""
import argparse
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[2] / "resultados"
SEED = 42


def dividir(dataset, seed=SEED):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(dataset))
    y = [s[1] for s in dataset.samples]
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    yr = [y[i] for i in rest]
    va, te = train_test_split(rest, test_size=0.50, stratify=yr, random_state=seed)
    return tr, va, te


def baseline_hog_svm(dataset, tr, te):
    from skimage.feature import hog
    from skimage.transform import resize
    from sklearn.svm import SVC
    from PIL import Image

    def feat(i):
        img = np.asarray(Image.open(dataset.samples[i][0]).convert("L"))
        return hog(resize(img, (128, 128)), pixels_per_cell=(16, 16))

    Xtr = np.array([feat(i) for i in tr]); ytr = [dataset.samples[i][1] for i in tr]
    Xte = np.array([feat(i) for i in te]); yte = [dataset.samples[i][1] for i in te]
    clf = SVC(kernel="rbf", C=10).fit(Xtr, ytr)
    return yte, clf.predict(Xte).tolist()


def main(a):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms as T
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from src.visao_computacional.classificador import construir_modelo, CLASSES

    torch.manual_seed(SEED); np.random.seed(SEED)
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    aug = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomRotation(15),
                     T.ColorJitter(brightness=0.2), T.ToTensor(), norm])
    sem = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])
    base = datasets.ImageFolder(a.dados)
    assert sorted(base.classes) == sorted(CLASSES), base.classes
    tr, va, te = dividir(base)
    dtr = datasets.ImageFolder(a.dados, transform=aug)
    dev = datasets.ImageFolder(a.dados, transform=sem)
    ltr = DataLoader(Subset(dtr, tr), batch_size=32, shuffle=True)
    lva = DataLoader(Subset(dev, va), batch_size=64)
    lte = DataLoader(Subset(dev, te), batch_size=64)

    m = construir_modelo(True)
    for p in m.features.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(m.classifier.parameters(), lr=1e-3)
    perda = nn.CrossEntropyLoss()
    hist = []
    for ep in range(a.epocas):
        if ep == a.epocas // 2:  # fine-tuning das últimas camadas
            for p in m.features[-4:].parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4)
        m.train(); tot = 0
        for x, y in ltr:
            opt.zero_grad(); l = perda(m(x), y); l.backward(); opt.step(); tot += l.item()
        m.eval(); ok = n = 0
        with torch.no_grad():
            for x, y in lva:
                ok += (m(x).argmax(1) == y).sum().item(); n += len(y)
        hist.append({"epoca": ep + 1, "perda": tot / len(ltr), "acc_val": ok / n})
        print(hist[-1])

    ys, ps = [], []
    m.eval()
    with torch.no_grad():
        for x, y in lte:
            ps += m(x).argmax(1).tolist(); ys += y.tolist()
    nomes = base.classes
    RES.mkdir(exist_ok=True)
    torch.save(m.state_dict(), RES / "mobilenetv2_trashnet.pt")
    yb, pb = baseline_hog_svm(base, tr, te)
    met = {
        "mobilenetv2": {"acuracia": accuracy_score(ys, ps), "f1_macro": f1_score(ys, ps, average="macro"),
                        "matriz": confusion_matrix(ys, ps).tolist()},
        "hog_svm": {"acuracia": accuracy_score(yb, pb), "f1_macro": f1_score(yb, pb, average="macro"),
                    "matriz": confusion_matrix(yb, pb).tolist()},
        "classes": nomes, "historico": hist,
        "tamanhos": {"treino": len(tr), "validacao": len(va), "teste": len(te)},
    }
    (RES / "metricas_imagens.json").write_text(json.dumps(met, indent=2, ensure_ascii=False))
    print({k: (v["acuracia"], v["f1_macro"]) for k, v in met.items() if isinstance(v, dict) and "acuracia" in v})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="dados/trashnet")
    ap.add_argument("--epocas", type=int, default=8)
    main(ap.parse_args())
