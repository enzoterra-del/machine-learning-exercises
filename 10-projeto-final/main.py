"""Exercício 10 - Projeto final: diagnóstico de câncer de mama.

Dataset real (Breast Cancer Wisconsin, vem com o scikit-learn).
Fluxo completo: exploração -> limpeza -> treino de vários modelos ->
validação cruzada -> otimização -> avaliação final -> conclusão.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

# 1. Dados e exploração -------------------------------------------------
dados = load_breast_cancer(as_frame=True)
df = dados.frame
print("Formato:", df.shape)
print("Classes:", dict(zip(dados.target_names, df["target"].value_counts().sort_index())))
print("Valores nulos:", int(df.isna().sum().sum()))
print("Linhas duplicadas:", int(df.duplicated().sum()), "\n")

# 2. Limpeza ------------------------------------------------------------
df = df.drop_duplicates().dropna()

# Variáveis muito correlacionadas entre si trazem informação repetida
X, y = df.drop(columns="target"), df["target"]
corr = X.corr().abs()
cols_remover = [c for i, c in enumerate(corr.columns)
                if (corr.iloc[:i][c] > 0.95).any()]
print(f"Removendo {len(cols_remover)} variáveis com correlação > 0.95\n")
X = X.drop(columns=cols_remover)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Comparação de modelos com validação cruzada --------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidatos = {
    "Regressão Logística": Pipeline([("esc", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=2000))]),
    "SVM": Pipeline([("esc", StandardScaler()), ("m", SVC(probability=True))]),
    "Random Forest": Pipeline([("m", RandomForestClassifier(random_state=42))]),
}
print("Validação cruzada (F1):")
resultados = {}
for nome, modelo in candidatos.items():
    scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1")
    resultados[nome] = scores.mean()
    print(f"  {nome:20s} {scores.mean():.4f} (+/- {scores.std():.4f})")

melhor_nome = max(resultados, key=resultados.get)
print(f"\nMelhor candidato: {melhor_nome}\n")

# 4. Otimização de hiperparâmetros ----------------------------------------
grades = {
    "Regressão Logística": {"m__C": [0.01, 0.1, 1, 10, 100]},
    "SVM": {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.01, 0.001]},
    "Random Forest": {"m__n_estimators": [100, 300], "m__max_depth": [4, 8, None]},
}
busca = GridSearchCV(candidatos[melhor_nome], grades[melhor_nome],
                     cv=cv, scoring="f1", n_jobs=-1).fit(X_train, y_train)
print("Melhores parâmetros:", busca.best_params_)
print("F1 (validação cruzada):", round(busca.best_score_, 4), "\n")

# 5. Avaliação final no conjunto de teste ---------------------------------
final = busca.best_estimator_
pred = final.predict(X_test)
proba = final.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred, target_names=dados.target_names))
print("ROC AUC:", round(roc_auc_score(y_test, proba), 4))
print("Recall da classe maligna:",
      round(recall_score(y_test, pred, pos_label=0), 4), "\n")

ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=dados.target_names)
plt.title(f"{melhor_nome}: matriz de confusão")
plt.savefig(FIGS / "matriz_confusao.png", dpi=120)
plt.close()

RocCurveDisplay.from_predictions(y_test, proba)
plt.title(f"{melhor_nome}: curva ROC")
plt.savefig(FIGS / "roc.png", dpi=120)
plt.close()

# 6. Conclusão ------------------------------------------------------------
print("Conclusão: o modelo otimizado generaliza bem, mas em diagnóstico médico")
print("o recall da classe maligna é a métrica mais crítica (falso negativo é")
print("mais grave que falso positivo). Este modelo é educacional e não substitui")
print("avaliação médica.")
