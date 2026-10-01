"""Exercício 01 - Regressão Linear: prever o preço de casas.

Os dados são sintéticos (gerados com semente fixa), então o script roda
offline e o resultado é reprodutível.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# 1. Dados -------------------------------------------------------------
n = 500
df = pd.DataFrame({
    "area_m2": rng.uniform(40, 300, n),
    "quartos": rng.integers(1, 6, n),
    "idade_anos": rng.integers(0, 50, n),
    "distancia_centro_km": rng.uniform(0.5, 25, n),
})
df["preco"] = (
    50_000
    + 3_000 * df["area_m2"]
    + 15_000 * df["quartos"]
    - 1_200 * df["idade_anos"]
    - 4_000 * df["distancia_centro_km"]
    + rng.normal(0, 30_000, n)
)

# 2. Exploração --------------------------------------------------------
print(df.head(), "\n")
print(df.describe().round(1), "\n")
print("Correlação com o preço:\n", df.corr()["preco"].drop("preco").round(3), "\n")

# 3. Treino / teste ----------------------------------------------------
X = df.drop(columns="preco")
y = df["preco"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = LinearRegression().fit(X_train, y_train)

# 4. Avaliação ---------------------------------------------------------
pred = modelo.predict(X_test)
print("MAE :", round(mean_absolute_error(y_test, pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
print("R²  :", round(r2_score(y_test, pred), 4), "\n")
print("Coeficientes:")
for nome, coef in zip(X.columns, modelo.coef_):
    print(f"  {nome:22s} {coef:12.2f}")

# 5. Gráfico -----------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.6)
lim = [y.min(), y.max()]
plt.plot(lim, lim, "r--", label="previsão perfeita")
plt.xlabel("Preço real")
plt.ylabel("Preço previsto")
plt.title("Regressão Linear: real vs previsto")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "real_vs_previsto.png", dpi=120)
plt.close()
