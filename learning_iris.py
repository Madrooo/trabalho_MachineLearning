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

print(df.head(8))
print(df.dtypes)
print("Tipo:", type(df), "| Linhas e colunas:", df.shape)
print(df.describe())
print(df["variety"].value_counts())

# explorando os dados
print("\nsepal.length > 5.8:", len(df[df["sepal.length"] > 5.8]))
print("sem nulo em sepal.length:", len(df[df["sepal.length"].notna()]))

df_copia = df.copy()   # cópia para a coluna nova não entrar no modelo
df_copia["sepal.length2"] = df_copia["sepal.length"] ** 2
print(df_copia[["sepal.length", "sepal.length2"]].head(3))

print("Media do sepal.length:", df["sepal.length"].mean())
print(df[["sepal.length", "petal.length"]].median())

# ----------------------------------------------------------------
# 2. Pré-processamento
# ----------------------------------------------------------------
print("\n2. PRE-PROCESSAMENTO")
print("Nulos:", df.isnull().sum().sum(), "| Duplicadas:", df.duplicated().sum())
df = df.drop_duplicates()

cols = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

# min-max manual: (x - min) / (max - min), só para ver como funciona
df_scaled = df.copy()
for col in cols:
    df_scaled[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
print(df_scaled[cols].agg(["min", "max", "count"]))
df_scaled.to_csv("iris_minmax.csv", index=False)

# características (X) e classe (y)
X = df[cols].copy()
y = df["variety"]

# ----------------------------------------------------------------
# 3. Divisão treino-teste (hold-out)
# ----------------------------------------------------------------
print("\n3. DIVISAO TREINO-TESTE")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = X_train.copy()
X_test = X_test.copy()
print("Treino:", X_train.shape, "| Teste:", X_test.shape)

# nulos (se tiver) preenchidos com a mediana do treino
medianas = X_train.median()
X_train = X_train.fillna(medianas)
X_test = X_test.fillna(medianas)

# min-max: o scaler aprende só com o treino (fit) e é aplicado nos dois
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

# exemplo da aula (1 = grávida, 0 = não grávida)
y_true_ex = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
y_pred_ex = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]

# labels=[1, 0] deixa a matriz igual ao slide: [[TP, FN], [FP, TN]]
cm_ex = confusion_matrix(y_true_ex, y_pred_ex, labels=[1, 0])
TP, FN, FP, TN = cm_ex.ravel()

acc = (TP + TN) / (TP + FP + TN + FN)
prec = TP / (TP + FP)
rec = TP / (TP + FN)
f1 = 2 * (prec * rec) / (prec + rec)
print("Exemplo da aula:", cm_ex.tolist())
print("Acuracia:", round(acc, 2), "| Precisao:", round(prec, 2),
      "| Recall:", round(rec, 2), "| F1:", round(f1, 2))

# resultado no Iris (linhas = valor real, colunas = valor predito)
classes = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=classes)
print("\nClasses:", classes)
print(cm)
print("Acuracia geral:", round(accuracy_score(y_test, y_pred), 4))

# no Iris são 3 classes: cada uma vira a "positiva" e as outras duas a "negativa"
for i in range(len(classes)):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - TP - FN - FP

    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    f1 = 2 * (prec * rec) / (prec + rec)
    print(classes[i], "-> TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN,
          "| Precisao:", round(prec, 4), "Recall:", round(rec, 4), "F1:", round(f1, 4))

# ----------------------------------------------------------------
# 6. Técnicas de validação
# ----------------------------------------------------------------
print("\n6. TECNICAS DE VALIDACAO")


# escala (só com o treino), treina o KNN e devolve a acurácia no teste
def treinar_e_avaliar(X_tr, X_te, y_tr, y_te):
    esc = MinMaxScaler()
    X_tr = esc.fit_transform(X_tr)
    X_te = esc.transform(X_te)
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_tr, y_tr)
    return accuracy_score(y_te, modelo.predict(X_te))


# hold-out: é a divisão treino-teste do passo 3
print("Hold-out:", round(accuracy_score(y_test, y_pred), 4))

# n-hold-out: repete o hold-out com divisões diferentes
print("\nn-hold-out (5 repeticoes):")
accs = []
for i in range(5):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=i, stratify=y
    )
    accs.append(treinar_e_avaliar(Xtr, Xte, ytr, yte))
    print("  Repeticao", i + 1, "->", round(accs[-1], 4))
print("Media:", round(sum(accs) / len(accs), 4))

# k-fold com 5 folds: em cada iteração 4 partes treinam e 1 testa
print("\nK-fold (5 folds):")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accs_kf = []
for n, (idx_treino, idx_teste) in enumerate(kf.split(X), start=1):
    acc_i = treinar_e_avaliar(X.iloc[idx_treino], X.iloc[idx_teste],
                              y.iloc[idx_treino], y.iloc[idx_teste])
    accs_kf.append(acc_i)
    print("  Iteracao", n, "-> treino:", len(idx_treino),
          "| teste:", len(idx_teste), "| acuracia:", round(acc_i, 4))
print("Media:", round(sum(accs_kf) / len(accs_kf), 4))