"""Exercício 06 - SVM: classificar imagens (dígitos manuscritos 8x8)."""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

digitos = load_digits()
X, y = digitos.data, digitos.target
print("Imagens:", X.shape[0], "| pixels por imagem:", X.shape[1], "\n")

# Algumas imagens de exemplo
fig, eixos = plt.subplots(2, 8, figsize=(10, 3))
for ax, img, rot in zip(eixos.ravel(), digitos.images, digitos.target):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(rot)
    ax.axis("off")
plt.savefig(FIGS / "amostras.png", dpi=120)
plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Busca de hiperparâmetros (kernel, C e gamma)
pipe = make_pipeline(StandardScaler(), SVC())
grade = {
    "svc__kernel": ["linear", "rbf"],
    "svc__C": [0.1, 1, 10],
    "svc__gamma": ["scale", 0.001, 0.01],
}
busca = GridSearchCV(pipe, grade, cv=5, n_jobs=-1).fit(X_train, y_train)
print("Melhores parâmetros:", busca.best_params_)
print("Acurácia na validação cruzada:", round(busca.best_score_, 4))

pred = busca.predict(X_test)
print("Acurácia no teste:", round(accuracy_score(y_test, pred), 4))

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("SVM: matriz de confusão (dígitos)")
plt.savefig(FIGS / "matriz_confusao.png", dpi=120)
plt.close()
