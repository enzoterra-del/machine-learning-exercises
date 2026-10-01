"""Exercício 02 - KNN: classificar espécies de flores (Iris)."""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
print(X.describe().round(2), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# KNN é baseado em distância, então precisa de escalonamento
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# Escolhendo o melhor k
ks = range(1, 21)
acuracias = []
for k in ks:
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
    acuracias.append(m.score(X_test_s, y_test))
melhor_k = ks[acuracias.index(max(acuracias))]
print(f"Melhor k: {melhor_k} (acurácia {max(acuracias):.3f})\n")

modelo = KNeighborsClassifier(n_neighbors=melhor_k).fit(X_train_s, y_train)
pred = modelo.predict(X_test_s)
print(classification_report(y_test, pred, target_names=iris.target_names))

plt.figure()
plt.plot(list(ks), acuracias, marker="o")
plt.xlabel("k")
plt.ylabel("Acurácia")
plt.title("KNN: acurácia por valor de k")
plt.savefig(FIGS / "acuracia_por_k.png", dpi=120)
plt.close()

ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=iris.target_names
)
plt.title("KNN: matriz de confusão")
plt.savefig(FIGS / "matriz_confusao.png", dpi=120)
plt.close()
