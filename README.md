# Machine Learning Exercises

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.1%2B-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-em%20andamento-yellow)

Repositório de estudos práticos de **Machine Learning**, com uma trilha progressiva que vai dos algoritmos clássicos até um projeto final completo, com dataset real, limpeza, treino, validação, métricas e otimização do modelo.

Cada exercício é independente e segue o mesmo padrão: **problema → exploração dos dados → treino → avaliação → conclusões**.

---

## Trilha de aprendizado

| #  | Algoritmo / Tema         | Problema prático                          | Tipo            | Status |
|----|--------------------------|-------------------------------------------|-----------------|--------|
| 01 | Regressão Linear         | Prever o preço de casas                   | Regressão       | ✅ |
| 02 | KNN                      | Classificar espécies de flores            | Classificação   | ✅ |
| 03 | Regressão Logística      | Prever aprovado ou reprovado              | Classificação   | ✅ |
| 04 | Árvore de Decisão        | Classificar clientes                      | Classificação   | ✅ |
| 05 | Random Forest            | Detectar fraude em transações             | Classificação   | ✅ |
| 06 | SVM                      | Classificar imagens                       | Classificação   | ✅ |
| 07 | K-Means                  | Separar clientes por comportamento        | Clusterização   | ✅ |
| 08 | PCA                      | Reduzir dimensões de um dataset           | Redução de dim. | ✅ |
| 09 | Pipeline                 | Fluxo completo de ML                      | Engenharia      | ✅ |
| 10 | **Projeto final**        | Dataset real de ponta a ponta             | Completo        | ✅ |

Todos os exercícios rodam offline: usam datasets que vêm com o scikit-learn ou dados sintéticos gerados com semente fixa.

---

## Estrutura do repositório

```
machine-learning-exercises/
├── 01-regressao-linear/
├── 02-knn/
├── 03-regressao-logistica/
├── 04-arvore-de-decisao/
├── 05-random-forest/
├── 06-svm/
├── 07-kmeans/
├── 08-pca/
├── 09-pipeline/
├── 10-projeto-final/
├── requirements.txt
├── .gitignore
└── README.md
```

Cada pasta de exercício contém, em geral:

```
NN-nome-do-exercicio/
├── main.py                 # script completo e comentado
└── figs/                   # gráficos gerados pelo script
```

---

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/enzoterra-del/machine-learning-exercises.git
cd machine-learning-exercises
```

### 2. Criar e ativar um ambiente virtual

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Rodar um exercício

```bash
cd 01-regressao-linear
python main.py
```

Os gráficos são salvos na pasta `figs/` de cada exercício.

---

## Resultados

| #  | Modelo                | Métrica principal no teste                         |
|----|-----------------------|----------------------------------------------------|
| 01 | Regressão Linear      | R² = 0,98 · RMSE ≈ 30,4 mil                        |
| 02 | KNN (k=1)             | Acurácia = 96,7%                                   |
| 03 | Regressão Logística   | Acurácia = 90% · ROC AUC = 0,955                   |
| 04 | Árvore de Decisão     | Acurácia = 73,8% (profundidade 4)                  |
| 05 | Random Forest         | Precisão 96% · Recall 50% na classe fraude · PR-AUC = 0,857 |
| 06 | SVM (RBF)             | Acurácia = 98% nos dígitos                         |
| 07 | K-Means               | k = 4 pelo silhouette (0,77)                       |
| 08 | PCA                   | 10 de 13 componentes mantêm 95% da variância      |
| 09 | Pipeline + Random Forest | F1 = 0,61 na classe minoritária                |
| 10 | Projeto final         | ROC AUC = 0,994 · Recall maligno = 97,6%           |

Os valores vêm da execução com as sementes fixas dos scripts e podem variar um pouco entre versões do scikit-learn.

---

## Metodologia

Em todos os exercícios, o fluxo de trabalho é o mesmo:

1. **Entendimento do problema**: o que se quer prever e por quê.
2. **Análise exploratória (EDA)**: `head`, `info`, `describe`, distribuições e correlações.
3. **Preparação dos dados**: tratamento de nulos, codificação e escalonamento.
4. **Separação treino/teste**: `train_test_split` com `random_state` fixo para reprodutibilidade.
5. **Treinamento** do modelo.
6. **Avaliação** com métricas adequadas ao problema:
   - Regressão: MAE, MSE, RMSE e R²
   - Classificação: acurácia, precisão, recall, F1-score e matriz de confusão
   - Clusterização: inércia e silhouette score
7. **Visualização** dos resultados com `matplotlib` e `seaborn`.
8. **Conclusões**: limitações e próximos passos.

---

## Projeto final

O exercício 10 reúne tudo o que foi aprendido em um projeto completo:

- Dataset real
- Limpeza e tratamento dos dados
- Treinamento de modelos
- Validação cruzada
- Métricas de desempenho
- Otimização de hiperparâmetros (`GridSearchCV` / `RandomizedSearchCV`)
- Comparação entre modelos e conclusão

---

## Tecnologias

- **Python 3.10+**
- **pandas** e **NumPy**: manipulação de dados
- **scikit-learn**: modelos, métricas e pipelines
- **matplotlib** e **seaborn**: visualização

---

## Autor

**Enzo Terra**
GitHub: [@enzoterra-del](https://github.com/enzoterra-del)

Sugestões e feedbacks são bem-vindos. Abra uma *issue* ou envie um *pull request*.

---

## Licença

Distribuído sob a licença MIT. Adicione um arquivo `LICENSE` ao repositório para formalizar.
