"""Exercício 04 - Árvore de Decisão: classificar clientes.

Dados sintéticos: o cliente é classificado como "premium", "regular" ou
"basico" conforme renda, compras e tempo de relacionamento.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

n = 800
df = pd.DataFrame({
    "renda_mensal": rng.normal(5000, 2000, n).clip(1200, None),
    "compras_ano": rng.poisson(12, n),
    "anos_cliente": rng.uniform(0, 10, n),
})
pontos = (
    df["renda_mensal"] / 1000
    + df["compras_ano"] * 0.5
    + df["anos_cliente"] * 0.6
    + rng.normal(0, 1.5, n)
)
df["perfil"] = pd.cut(
    pontos, bins=[-np.inf, 10, 15, np.inf], labels=["basico", "regular", "premium"]
)
print(df["perfil"].value_counts(), "\n")

X, y = df.drop(columns="perfil"), df["perfil"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# max_depth limita o tamanho da árvore e evita overfitting
for prof in [2, 3, 5, 8, None]:
    m = DecisionTreeClassifier(max_depth=prof, random_state=42).fit(X_train, y_train)
    print(f"max_depth={str(prof):5s} treino={m.score(X_train, y_train):.3f} "
          f"teste={m.score(X_test, y_test):.3f}")

modelo = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
pred = modelo.predict(X_test)
print("\nModelo final (max_depth=4)")
print("Acurácia:", round(accuracy_score(y_test, pred), 3), "\n")
print(classification_report(y_test, pred))
print("Importância das variáveis:")
for nome, imp in zip(X.columns, modelo.feature_importances_):
    print(f"  {nome:15s} {imp:.3f}")

plt.figure(figsize=(16, 8))
plot_tree(modelo, feature_names=X.columns, class_names=modelo.classes_,
          filled=True, rounded=True, fontsize=8)
plt.savefig(FIGS / "arvore.png", dpi=120)
plt.close()
