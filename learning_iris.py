# Trabalho de Machine Learning - KNN com o dataset Iris

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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
print("Tipo do objeto:", type(df))
print("Linhas e colunas:", df.shape)
print()
print(df.describe())
print()

# a coluna variety é a espécie da flor (o que queremos prever)
print(df["variety"].value_counts())
print()

# --- explorando os dados ---

# filtrando linhas
acima_58 = df[df["sepal.length"] > 5.8]
print("Flores com sepal.length > 5.8:", len(acima_58))
print(acima_58.head())
print()

# True/False para cada linha
print((df["sepal.length"] > 5.8).head())
print()

# linhas sem valor nulo na coluna
sem_na = df[df["sepal.length"].notna()]
print("Linhas sem nulo em sepal.length:", len(sem_na))
print(sem_na.head())
print()

# criando uma coluna nova (valor ao quadrado)
# faço em uma cópia para a coluna nova não entrar no modelo
df_copia = df.copy()
df_copia["sepal.length2"] = df_copia["sepal.length"] ** 2
print(df_copia[["sepal.length", "sepal.length2"]].head())
print()

# média e mediana
print("Media do sepal.length:", df["sepal.length"].mean())
print(df[["sepal.length", "petal.length"]].median())

# ----------------------------------------------------------------
# 2. Pré-processamento
# ----------------------------------------------------------------
print("\n2. PRE-PROCESSAMENTO")

# verificando se tem valores nulos e linhas repetidas
print("Valores nulos por coluna:")
print(df.isnull().sum())
print("Linhas duplicadas:", df.duplicated().sum())

# remove as repetidas
# (se existirem nulos, eu preencho depois da divisão com a mediana do treino)
df = df.drop_duplicates()

# colunas numéricas que vão ser usadas como características
cols = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

# min-max manual: (x - min) / (max - min)
# só para ver como funciona, o modelo usa o MinMaxScaler mais abaixo
df_scaled = df.copy()
for col in cols:
    min_val = df[col].min()
    max_val = df[col].max()
    df_scaled[col] = (df[col] - min_val) / (max_val - min_val)

print("\nDepois do min-max manual:")
print(df_scaled[cols].agg(["min", "max", "count"]))

# salvando o arquivo novo
df_scaled.to_csv("iris_minmax.csv", index=False)

# separando as características (X) da classe (y)
features = cols
X = df[features].copy()
y = df["variety"]

# ----------------------------------------------------------------
# 3. Divisão treino-teste (hold-out)
# ----------------------------------------------------------------
print("\n3. DIVISAO TREINO-TESTE")

# 80% para treino e 20% para teste
# stratify=y mantém a mesma proporção de cada espécie nos dois conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = X_train.copy()
X_test = X_test.copy()
print("Tamanho do treino:", X_train.shape)
print("Tamanho do teste:", X_test.shape)

# preenchendo nulos (se tiver) com a mediana do treino
medianas = X_train.median()
X_train = X_train.fillna(medianas)
X_test = X_test.fillna(medianas)

# o KNN usa distância, então coloco tudo na mesma escala (0 a 1)
# o scaler aprende só com o treino (fit) e depois é aplicado no treino e no teste
scaler = MinMaxScaler()
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.transform(X_test[cols])

# ----------------------------------------------------------------
# 4. KNN
# ----------------------------------------------------------------
print("\n4. KNN")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Modelo treinado com k = 5")

# ----------------------------------------------------------------
# 5. Avaliação dos resultados
# ----------------------------------------------------------------
print("\n5. AVALIACAO DOS RESULTADOS")

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
print("Acuracia:", round(acc, 2))
print("Precisao:", round(prec, 2))
print("Recall:", round(rec, 2))
print("F1:", round(f1, 2))

# --- resultado do KNN no Iris ---
print("\nResultado do KNN no Iris:")

# linhas = valor real, colunas = valor predito
classes = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=classes)
print("Ordem das classes:", classes)
print("Matriz de confusao:")
print(cm)

print("\nAcuracia geral:", round(accuracy_score(y_test, y_pred), 4))

# no Iris são 3 classes, então calculo TP, FP, FN e TN de cada uma
# (a classe vira a "positiva" e as outras duas viram "negativa")
# e depois uso as mesmas fórmulas do slide
for i in range(len(classes)):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - TP - FN - FP

    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    f1 = 2 * (prec * rec) / (prec + rec)

    print("\nClasse:", classes[i])
    print("TP =", TP, "| FN =", FN, "| FP =", FP, "| TN =", TN)
    print("Precisao:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1:", round(f1, 4))

# ----------------------------------------------------------------
# 6. Técnicas de validação
# ----------------------------------------------------------------
print("\n6. TECNICAS DE VALIDACAO")

# --- hold-out: é a divisão treino-teste que já fiz acima ---
acc_holdout = accuracy_score(y_test, y_pred)
print("\nHold-out (uma divisao):", round(acc_holdout, 4))

# --- n-hold-out: repetir o hold-out várias vezes com divisões diferentes ---
print("\nn-hold-out (5 repeticoes):")
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
    print("  Repeticao", i + 1, "->", round(acc_i, 4))

media_nholdout = sum(resultados) / len(resultados)
print("Media:", round(media_nholdout, 4))

# --- k-fold cross validation com 5 folds ---
# os dados são divididos em 5 partes: em cada iteração 4 treinam e 1 testa
print("\nK-fold com 5 folds:")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
resultados_kf = []
iteracao = 1

for idx_treino, idx_teste in kf.split(X):
    X_tr = X.iloc[idx_treino]
    X_te = X.iloc[idx_teste]
    y_tr = y.iloc[idx_treino]
    y_te = y.iloc[idx_teste]

    # o scaler aprende só com o treino de cada iteração
    esc = MinMaxScaler()
    X_tr = esc.fit_transform(X_tr)
    X_te = esc.transform(X_te)

    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_tr, y_tr)
    acc_i = accuracy_score(y_te, modelo.predict(X_te))
    resultados_kf.append(acc_i)
    print("  Iteracao", iteracao, "-> treino:", len(idx_treino),
          "| teste:", len(idx_teste), "| acuracia:", round(acc_i, 4))
    iteracao = iteracao + 1

media_kf = sum(resultados_kf) / len(resultados_kf)
print("Media:", round(media_kf, 4))