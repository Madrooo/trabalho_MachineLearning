# Trabalho de Machine Learning - KNN com o dataset Iris

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------------------
# 1. Leitura dos dados
# ----------------------------------------------------------------
print("1. LEITURA DOS DADOS")

df = pd.read_csv("iris.csv")

# se o arquivo tiver uma coluna Id eu tiro, porque ela não ajuda no modelo
if "Id" in df.columns:
    df = df.drop(columns="Id")

print(df.head(8))
print()
print(df.dtypes)
print()
print("Linhas e colunas:", df.shape)
print()
print(df.describe())
print()

# a última coluna é a espécie da flor (o que queremos prever)
print(df.iloc[:, -1].value_counts())

# ----------------------------------------------------------------
# 2. Pré-processamento
# ----------------------------------------------------------------
print("\n2. PRÉ-PROCESSAMENTO")

# verificando se tem valores nulos e linhas repetidas
print("Valores nulos por coluna:")
print(df.isnull().sum())
print("Linhas duplicadas:", df.duplicated().sum())

df = df.dropna()
df = df.drop_duplicates()

# separando as características (X) da classe (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ----------------------------------------------------------------
# 3. Divisão treino-teste (hold-out)
# ----------------------------------------------------------------
print("\n3. DIVISÃO TREINO-TESTE")

# 80% para treino e 20% para teste
# stratify=y mantém a mesma proporção de cada espécie nos dois conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Tamanho do treino:", X_train.shape)
print("Tamanho do teste:", X_test.shape)

# o KNN usa distância, então coloco tudo na mesma escala (0 a 1)
# o scaler aprende só com o treino (fit) e depois é aplicado no treino e no teste
scaler = MinMaxScaler()
X_train_esc = scaler.fit_transform(X_train)
X_test_esc = scaler.transform(X_test)

# ----------------------------------------------------------------
# 4. KNN
# ----------------------------------------------------------------
print("\n4. KNN")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_esc, y_train)

y_pred = knn.predict(X_test_esc)
print("Modelo treinado com k = 5")

# ----------------------------------------------------------------
# 5. Avaliação dos resultados
# ----------------------------------------------------------------
print("\n5. AVALIAÇÃO DOS RESULTADOS")

# --- exemplo da aula (grávida = 1, não grávida = 0) ---
print("\nExemplo da aula:")
y_true_ex = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
y_pred_ex = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]

# labels=[1, 0] para a matriz ficar igual ao slide: [[TP, FN], [FP, TN]]
cm_ex = confusion_matrix(y_true_ex, y_pred_ex, labels=[1, 0])
print(cm_ex)

TP, FN, FP, TN = cm_ex.ravel()
print("TP =", TP, "| FN =", FN, "| FP =", FP, "| TN =", TN)

# fórmulas do slide
acc = (TP + TN) / (TP + FP + TN + FN)
prec = TP / (TP + FP)
rec = TP / (TP + FN)
f1 = 2 * (prec * rec) / (prec + rec)
print("Acurácia:", round(acc, 2))
print("Precisão:", round(prec, 2))
print("Recall:", round(rec, 2))
print("F1:", round(f1, 2))

# --- resultado do KNN no Iris ---
print("\nResultado do KNN no Iris:")

# linhas = valor real, colunas = valor predito
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

# como são 3 classes uso average="macro" (média das 3 classes)
print("Acurácia:", round(accuracy_score(y_test, y_pred), 4))
print("Precisão:", round(precision_score(y_test, y_pred, average="macro"), 4))
print("Recall:", round(recall_score(y_test, y_pred, average="macro"), 4))
print("F1:", round(f1_score(y_test, y_pred, average="macro"), 4))

print()
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------
# 6. Técnicas de validação
# ----------------------------------------------------------------
print("6. TÉCNICAS DE VALIDAÇÃO")

# --- hold-out: é a divisão treino-teste que já fiz acima ---
acc_holdout = accuracy_score(y_test, y_pred)
print("\nHold-out (uma divisão):", round(acc_holdout, 4))

# --- n-hold-out: repetir o hold-out várias vezes com divisões diferentes ---
print("\nn-hold-out (5 repetições):")
resultados = []
for i in range(5):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=i, stratify=y
    )
    esc = MinMaxScaler()
    Xtr = esc.fit_transform(Xtr)
    Xte = esc.transform(Xte)

    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(Xtr, ytr)
    acc_i = accuracy_score(yte, modelo.predict(Xte))
    resultados.append(acc_i)
    print("  Repetição", i + 1, "->", round(acc_i, 4))

media_nholdout = sum(resultados) / len(resultados)
print("Média:", round(media_nholdout, 4))

# --- k-fold cross validation com 5 folds ---
# uso o pipeline para o scaler ser refeito dentro de cada fold
print("\nK-fold com 5 folds:")
modelo_kfold = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=5))
scores = cross_val_score(modelo_kfold, X, y, cv=5)
print("Acurácia em cada fold:", scores.round(4))
print("Média:", round(scores.mean(), 4))

# --- testando valores diferentes de k com o k-fold ---
print("\nTestando valores de k:")
for k in [1, 3, 5, 7, 9, 11]:
    modelo_k = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=k))
    media = cross_val_score(modelo_k, X, y, cv=5).mean()
    print("  k =", k, "-> acurácia média =", round(media, 4))