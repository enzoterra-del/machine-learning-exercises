"""Exercício 05 - Random Forest: detectar fraude em transações.

Dados sintéticos e muito desbalanceados (cerca de 2% de fraudes), como
acontece na vida real. Por isso a acurácia sozinha engana: usamos
precisão, recall, F1 e PR-AUC.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (PrecisionRecallDisplay, average_precision_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

nomes = ["valor", "hora", "dist_ultima_compra", "n_transacoes_24h",
         "idade_conta", "score_dispositivo", "pais_risco", "tentativas_senha"]
X, y = make_classification(
    n_samples=20_000, n_features=8, n_informative=5, n_redundant=1,
    weights=[0.98, 0.02], flip_y=0.002, class_sep=1.2, random_state=42,
)
X = pd.DataFrame(X, columns=nomes)
print("Fraudes:", y.sum(), f"({y.mean():.2%})\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# class_weight="balanced" dá mais peso à classe rara
modelo = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
).fit(X_train, y_train)

pred = modelo.predict(X_test)
proba = modelo.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred, target_names=["legítima", "fraude"]))
print("Matriz de confusão:\n", confusion_matrix(y_test, pred))
print("PR-AUC:", round(average_precision_score(y_test, proba), 3), "\n")

imp = pd.Series(modelo.feature_importances_, index=nomes).sort_values()
print("Importância das variáveis:\n", imp.round(3).to_string())

imp.plot.barh(title="Random Forest: importância das variáveis")
plt.tight_layout()
plt.savefig(FIGS / "importancia.png", dpi=120)
plt.close()

PrecisionRecallDisplay.from_predictions(y_test, proba)
plt.title("Curva Precisão x Recall")
plt.savefig(FIGS / "precisao_recall.png", dpi=120)
plt.close()
