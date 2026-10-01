"""Exercício 03 - Regressão Logística: prever aprovado ou reprovado.

Dados sintéticos: a chance de aprovação cresce com as horas de estudo e
com a frequência, e cai com as faltas.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

n = 600
df = pd.DataFrame({
    "horas_estudo": rng.uniform(0, 10, n),
    "frequencia_pct": rng.uniform(40, 100, n),
    "nota_prova_anterior": rng.uniform(0, 10, n),
})
score = (
    0.6 * df["horas_estudo"]
    + 0.05 * df["frequencia_pct"]
    + 0.4 * df["nota_prova_anterior"]
    - 7
    + rng.normal(0, 1, n)
)
df["aprovado"] = (1 / (1 + np.exp(-score)) > 0.5).astype(int)

print(df.head(), "\n")
print("Proporção de aprovados:", df["aprovado"].mean().round(3), "\n")

X, y = df.drop(columns="aprovado"), df["aprovado"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = LogisticRegression(max_iter=1000).fit(X_train, y_train)
pred = modelo.predict(X_test)
proba = modelo.predict_proba(X_test)[:, 1]

print("Acurácia:", round(accuracy_score(y_test, pred), 3))
print("ROC AUC :", round(roc_auc_score(y_test, proba), 3), "\n")
print(classification_report(y_test, pred, target_names=["reprovado", "aprovado"]))

# Exemplo de uso: um aluno novo
aluno = pd.DataFrame([{"horas_estudo": 6, "frequencia_pct": 85,
                       "nota_prova_anterior": 7}])
print(f"Chance de aprovação do aluno exemplo: {modelo.predict_proba(aluno)[0, 1]:.1%}")

ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["reprovado", "aprovado"]
)
plt.title("Regressão Logística: matriz de confusão")
plt.savefig(FIGS / "matriz_confusao.png", dpi=120)
plt.close()
