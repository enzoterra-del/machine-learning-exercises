"""Exercício 09 - Pipeline: fluxo completo de ML.

Dados sintéticos com colunas numéricas e categóricas e valores ausentes.
O Pipeline junta imputação, escalonamento, one-hot encoding e modelo em
um único objeto, o que evita vazamento de dados (data leakage).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "idade": rng.integers(18, 70, n).astype(float),
    "salario": rng.normal(4500, 1500, n),
    "tempo_empresa": rng.uniform(0, 20, n),
    "cargo": rng.choice(["analista", "gerente", "estagiario", "diretor"], n),
    "cidade": rng.choice(["SP", "RJ", "BH", "POA"], n),
})
risco = (
    -0.0004 * (df["salario"] - 4500)
    - 0.1 * df["tempo_empresa"]
    + (df["cargo"] == "estagiario") * 1.0
    + rng.normal(0, 1, n)
)
df["saiu_da_empresa"] = (risco > 0.3).astype(int)

# Introduz valores ausentes
for col in ["idade", "salario", "cargo"]:
    df.loc[rng.choice(n, 60, replace=False), col] = np.nan

X, y = df.drop(columns="saiu_da_empresa"), df["saiu_da_empresa"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numericas = ["idade", "salario", "tempo_empresa"]
categoricas = ["cargo", "cidade"]

preprocessamento = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("esc", StandardScaler())]), numericas),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categoricas),
])

pipe = Pipeline([
    ("prep", preprocessamento),
    ("modelo", RandomForestClassifier(class_weight="balanced", random_state=42)),
])

busca = GridSearchCV(
    pipe,
    {"modelo__n_estimators": [100, 200], "modelo__max_depth": [4, 8, None]},
    cv=5, scoring="f1", n_jobs=-1,
).fit(X_train, y_train)

print("Melhores parâmetros:", busca.best_params_)
print("F1 na validação cruzada:", round(busca.best_score_, 3), "\n")
print(classification_report(y_test, busca.predict(X_test)))

# Salva o pipeline inteiro e carrega para usar em dados novos (com NaN!)
caminho = Path(__file__).parent / "modelo_pipeline.joblib"
joblib.dump(busca.best_estimator_, caminho)
modelo = joblib.load(caminho)
novo = pd.DataFrame([{"idade": np.nan, "salario": 2500, "tempo_empresa": 0.5,
                      "cargo": "estagiario", "cidade": "SP"}])
print("Previsão para um funcionário novo (com idade ausente):",
      modelo.predict(novo)[0])
