"""
Trabalho de Machine Learning - KNN com o dataset Iris

Etapas:
  1. Leitura dos dados (iris.csv)
  2. Pré-processamento
  3. Divisão treino-teste
  4. Treinamento do KNN
  5. Avaliação dos resultados (métricas + validação cruzada)

Uso:
  python trabalho_knn_iris.py              # procura iris.csv na mesma pasta
  python trabalho_knn_iris.py caminho.csv  # ou informe o caminho do arquivo
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

SEED = 42

# ----------------------------------------------------------------------
# 1. LEITURA DOS DADOS
# ----------------------------------------------------------------------
caminho = sys.argv[1] if len(sys.argv) > 1 else "iris.csv"
df = pd.read_csv(caminho)

print("=" * 60)
print("1. LEITURA DOS DADOS")
print("=" * 60)
print(f"Arquivo: {caminho}")
print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas\n")
print("Primeiras linhas:")
print(df.head(), "\n")
print("Informações gerais:")
df.info()
print("\nEstatísticas descritivas:")
print(df.describe(), "\n")

# ----------------------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO
# ----------------------------------------------------------------------
print("=" * 60)
print("2. PRÉ-PROCESSAMENTO")
print("=" * 60)

# Algumas versões do iris.csv trazem uma coluna de identificação (Id)
for col in df.columns:
    if col.strip().lower() in ("id", "unnamed: 0"):
        df = df.drop(columns=col)
        print(f"Coluna de identificação removida: {col}")

# Valores ausentes e duplicados
print(f"Valores ausentes por coluna:\n{df.isnull().sum()}\n")
df = df.dropna()
duplicadas = df.duplicated().sum()
print(f"Linhas duplicadas: {duplicadas}")
df = df.drop_duplicates().reset_index(drop=True)

# A última coluna é a classe (espécie); as demais são as características
coluna_alvo = df.columns[-1]
X = df.drop(columns=coluna_alvo).to_numpy(dtype=float)
y_texto = df[coluna_alvo].to_numpy()

print(f"Coluna alvo: {coluna_alvo}")
print(f"Características: {list(df.columns[:-1])}")
print(f"Distribuição das classes:\n{df[coluna_alvo].value_counts()}\n")

# Codifica as classes (texto -> número)
codificador = LabelEncoder()
y = codificador.fit_transform(y_texto)
nomes_classes = codificador.classes_
print(f"Classes codificadas: {dict(zip(nomes_classes, range(len(nomes_classes))))}\n")

# ----------------------------------------------------------------------
# 3. DIVISÃO TREINO-TESTE
# ----------------------------------------------------------------------
print("=" * 60)
print("3. DIVISÃO TREINO-TESTE")
print("=" * 60)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Treino: {X_treino.shape[0]} amostras | Teste: {X_teste.shape[0]} amostras\n")

# O KNN é baseado em distância, então as características precisam estar na
# mesma escala. O scaler é ajustado SÓ no treino para evitar vazamento de dados.
scaler = StandardScaler()
X_treino_esc = scaler.fit_transform(X_treino)
X_teste_esc = scaler.transform(X_teste)

# ----------------------------------------------------------------------
# 4. KNN
# ----------------------------------------------------------------------
print("=" * 60)
print("4. TREINAMENTO DO KNN")
print("=" * 60)

# Escolha do melhor k por validação cruzada (usando apenas o treino)
validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
valores_k = np.arange(1, 21)
medias_cv = []

for k in valores_k:
    modelo_k = KNeighborsClassifier(n_neighbors=int(k))
    scores = cross_val_score(modelo_k, X_treino_esc, y_treino, cv=validacao)
    medias_cv.append(scores.mean())

medias_cv = np.array(medias_cv)
melhor_k = int(valores_k[np.argmax(medias_cv)])
print(f"Melhor k (validação cruzada): {melhor_k} "
      f"(acurácia média = {medias_cv.max():.4f})\n")

modelo = KNeighborsClassifier(n_neighbors=melhor_k)
modelo.fit(X_treino_esc, y_treino)

# ----------------------------------------------------------------------
# 5. AVALIAÇÃO DOS RESULTADOS
# ----------------------------------------------------------------------
print("=" * 60)
print("5. AVALIAÇÃO DOS RESULTADOS")
print("=" * 60)

y_pred = modelo.predict(X_teste_esc)

print(f"Acurácia no teste: {accuracy_score(y_teste, y_pred):.4f}\n")
print("Relatório de classificação (precisão, recall, f1-score):")
print(classification_report(y_teste, y_pred, target_names=nomes_classes))

print("Matriz de confusão:")
print(confusion_matrix(y_teste, y_pred), "\n")

# Validação cruzada do modelo final (dados completos, com scaler dentro do fluxo)
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=melhor_k))
scores_finais = cross_val_score(pipeline, X, y, cv=validacao)
print(f"Validação cruzada (5 folds) no conjunto completo: {np.round(scores_finais, 4)}")
print(f"Média = {scores_finais.mean():.4f} | Desvio padrão = {scores_finais.std():.4f}")

# ----------------------------------------------------------------------
# GRÁFICOS
# ----------------------------------------------------------------------
fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))

eixos[0].plot(valores_k, medias_cv, marker="o")
eixos[0].axvline(melhor_k, color="red", linestyle="--", label=f"melhor k = {melhor_k}")
eixos[0].set_xlabel("k (número de vizinhos)")
eixos[0].set_ylabel("Acurácia média (validação cruzada)")
eixos[0].set_title("Escolha do k")
eixos[0].set_xticks(valores_k)
eixos[0].legend()

ConfusionMatrixDisplay.from_predictions(
    y_teste, y_pred, display_labels=nomes_classes, ax=eixos[1], colorbar=False
)
eixos[1].set_title("Matriz de confusão (teste)")

plt.tight_layout()
plt.savefig("resultados_knn.png", dpi=150)
print("\nGráficos salvos em resultados_knn.png")