"""Exercício 08 - PCA: reduzir dimensões de um dataset (vinhos, 13 variáveis)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

vinho = load_wine(as_frame=True)
X, y = vinho.data, vinho.target
print("Formato original:", X.shape, "\n")

X_s = StandardScaler().fit_transform(X)

# Variância explicada por componente
pca = PCA().fit(X_s)
acum = np.cumsum(pca.explained_variance_ratio_)
for i, v in enumerate(acum[:6], 1):
    print(f"{i} componente(s): {v:.1%} da variância")
n_95 = int(np.argmax(acum >= 0.95)) + 1
print(f"\nComponentes necessários para 95% da variância: {n_95} (de {X.shape[1]})\n")

# Projeção em 2D
X_2d = PCA(n_components=2).fit_transform(X_s)
plt.figure(figsize=(7, 5))
for classe, nome in enumerate(vinho.target_names):
    pts = X_2d[y == classe]
    plt.scatter(pts[:, 0], pts[:, 1], label=nome, alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA: vinhos projetados em 2 dimensões")
plt.legend()
plt.savefig(FIGS / "projecao_2d.png", dpi=120)
plt.close()

plt.figure()
plt.plot(range(1, len(acum) + 1), acum, marker="o")
plt.axhline(0.95, color="r", linestyle="--")
plt.xlabel("Número de componentes")
plt.ylabel("Variância acumulada")
plt.title("PCA: variância explicada")
plt.savefig(FIGS / "variancia.png", dpi=120)
plt.close()

# O modelo perde muito ao reduzir dimensões?
for n_comp in [2, n_95, X.shape[1]]:
    pipe = make_pipeline(StandardScaler(), PCA(n_components=n_comp),
                         LogisticRegression(max_iter=1000))
    acc = cross_val_score(pipe, X, y, cv=5).mean()
    print(f"Regressão Logística com {n_comp:2d} componentes: acurácia CV = {acc:.3f}")
