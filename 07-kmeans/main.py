"""Exercício 07 - K-Means: separar clientes por comportamento.

Dados sintéticos com 4 perfis escondidos. O K-Means não conhece os
rótulos: ele precisa descobrir os grupos sozinho.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

X, _ = make_blobs(n_samples=600, centers=4, n_features=2,
                  cluster_std=1.1, random_state=42)
# Reescala para unidades realistas
df = pd.DataFrame({
    "gasto_mensal": X[:, 0] * 120 + 1500,
    "visitas_mes": X[:, 1] * 1.5 + 20,
})
print(df.describe().round(1), "\n")

X_s = StandardScaler().fit_transform(df)

# Método do cotovelo e silhouette para escolher k
ks = range(2, 10)
inercias, silhuetas = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_s)
    inercias.append(km.inertia_)
    silhuetas.append(silhouette_score(X_s, km.labels_))
    print(f"k={k}  inércia={km.inertia_:8.1f}  silhouette={silhuetas[-1]:.3f}")

melhor_k = list(ks)[silhuetas.index(max(silhuetas))]
print(f"\nMelhor k pelo silhouette: {melhor_k}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(list(ks), inercias, marker="o")
a1.set(title="Método do cotovelo", xlabel="k", ylabel="Inércia")
a2.plot(list(ks), silhuetas, marker="o", color="green")
a2.set(title="Silhouette score", xlabel="k")
plt.tight_layout()
plt.savefig(FIGS / "escolha_k.png", dpi=120)
plt.close()

modelo = KMeans(n_clusters=melhor_k, n_init=10, random_state=42).fit(X_s)
df["grupo"] = modelo.labels_
print("\nPerfil médio de cada grupo:")
print(df.groupby("grupo").mean().round(1))
print("\nClientes por grupo:\n", df["grupo"].value_counts().sort_index())

plt.figure(figsize=(7, 5))
plt.scatter(df["gasto_mensal"], df["visitas_mes"], c=df["grupo"], cmap="viridis", s=18)
plt.xlabel("Gasto mensal")
plt.ylabel("Visitas por mês")
plt.title(f"K-Means: {melhor_k} grupos de clientes")
plt.savefig(FIGS / "grupos.png", dpi=120)
plt.close()
