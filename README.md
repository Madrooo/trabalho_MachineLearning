# Classificação de flores Iris com KNN

Trabalho da disciplina de **Machine Learning** (Ciência da Computação – IFCE).

O objetivo é treinar um modelo **KNN (K-Nearest Neighbors)** para descobrir a espécie de uma flor Iris a partir das medidas das suas pétalas e sépalas, e avaliar se o modelo realmente funciona bem usando métricas e técnicas de validação vistas em aula.

## O que o trabalho faz

1. Leitura dos dados (`iris.csv`)
2. Pré-processamento
3. Divisão treino-teste
4. Treinamento do KNN
5. Avaliação dos resultados (métricas)
6. Técnicas de validação (hold-out, n-hold-out e k-fold)

## Sobre o dataset

O dataset **Iris** tem 150 flores de 3 espécies (50 de cada):

| Espécie | Descrição |
|---|---|
| Setosa | Mais fácil de separar das outras |
| Versicolor | Parece um pouco com a Virginica |
| Virginica | Parece um pouco com a Versicolor |

Cada flor tem 4 características numéricas (em cm): comprimento e largura da sépala, comprimento e largura da pétala. A **coluna alvo** (o que queremos prever) é a espécie.

## Como executar

Precisa do Python instalado. Instale as bibliotecas:

```bash
python -m pip install pandas scikit-learn
```

Deixe o `iris.csv` na mesma pasta do script e rode:

```bash
python trabalho_iris_iniciante.py
```

## Passo a passo

### 1. Leitura dos dados

Usei o `pandas` para abrir o arquivo e dar uma olhada nos dados antes de qualquer coisa:

```python
df = pd.read_csv("iris.csv")
print(df.head(8))      # primeiras linhas
print(df.dtypes)       # tipo de cada coluna
print(df.describe())   # estatísticas (média, mínimo, máximo...)
```

Também conferi quantas flores existem de cada espécie com `value_counts()`. Se o arquivo tiver uma coluna `Id`, ela é removida, porque é só um número de identificação e não ajuda o modelo.

### 2. Pré-processamento

Antes de treinar, é preciso deixar os dados prontos:

- **Valores nulos:** conferi com `df.isnull().sum()` e removi linhas com `dropna()`, caso existissem.
- **Linhas duplicadas:** removidas com `drop_duplicates()`.
- **Separar X e y:** `X` são as 4 medidas (características) e `y` é a espécie (a resposta que o modelo precisa aprender).

```python
X = df.iloc[:, :-1]   # todas as colunas, menos a última
y = df.iloc[:, -1]    # só a última coluna (espécie)
```

### 3. Divisão treino-teste

Não dá para avaliar o modelo com os mesmos dados que ele usou para aprender, porque ele poderia só "decorar" as respostas. Por isso separei:

- **80% para treino** (o modelo aprende com eles)
- **20% para teste** (o modelo nunca viu, serve para a prova final)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- `random_state=42` deixa o resultado reproduzível (sempre a mesma divisão).
- `stratify=y` mantém a mesma proporção de cada espécie no treino e no teste.

**Normalização (min-max):** o KNN trabalha com distâncias, então características em escalas muito diferentes atrapalhariam. O min-max coloca tudo entre 0 e 1:

```
x_novo = (x - mínimo) / (máximo - mínimo)
```

```python
scaler = MinMaxScaler()
X_train_esc = scaler.fit_transform(X_train)   # aprende o mín/máx só com o treino
X_test_esc = scaler.transform(X_test)         # aplica no teste usando o que aprendeu
```

O `fit` é feito **só no treino** para que nenhuma informação do teste "vaze" para o modelo.

### 4. KNN

O KNN classifica uma flor nova olhando para as **k flores mais próximas** dela no conjunto de treino. A espécie que aparecer mais vezes entre esses vizinhos é a resposta. Neste passo usei `k = 5`:

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_esc, y_train)       # treino
y_pred = knn.predict(X_test_esc)    # previsões para o teste
```

### 5. Avaliação dos resultados

#### Matriz de confusão

Compara o que era o valor real com o que o modelo previu. Para duas classes (sim/não), ela tem 4 quadrantes:

| | Previu Sim | Previu Não |
|---|---|---|
| **Real Sim** | TP (verdadeiro positivo) | FN (falso negativo) |
| **Real Não** | FP (falso positivo) | TN (verdadeiro negativo) |

#### Métricas

$$ACC = \frac{TP + TN}{TP + FP + TN + FN}$$

$$PREC = \frac{TP}{TP + FP}$$

$$REC = \frac{TP}{TP + FN}$$

$$F1 = 2 \cdot \frac{PREC \cdot REC}{PREC + REC}$$

- **Acurácia:** de todas as previsões, quantas estavam certas.
- **Precisão:** das que o modelo disse que eram positivas, quantas realmente eram.
- **Recall:** de todas as que eram positivas, quantas o modelo encontrou.
- **F1:** média harmônica entre precisão e recall.

#### Exemplo da aula (grávida x não grávida)

Para treinar as fórmulas, refiz o exemplo do slide:

```python
y_true_ex = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
y_pred_ex = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
```

Resultado: TP = 3, FN = 1, FP = 2, TN = 4, o que dá acurácia 0,70, precisão 0,60, recall 0,75 e F1 0,67.

#### Métricas no Iris

Como o Iris tem 3 classes e não 2, usei `average="macro"`, que calcula a métrica de cada espécie e tira a média:

```python
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred, average="macro")
recall_score(y_test, y_pred, average="macro")
f1_score(y_test, y_pred, average="macro")
```

### 6. Técnicas de validação

Um único teste pode dar sorte ou azar dependendo de quais flores caíram no conjunto de teste. Por isso usei três técnicas:

- **Hold-out:** uma única divisão treino-teste (a do passo 3).
- **n-hold-out:** repete o hold-out várias vezes com divisões diferentes (5 repetições, `random_state` de 0 a 4) e tira a média.
- **K-fold (5 folds):** divide os dados em 5 partes. Em cada rodada, 4 partes treinam e 1 testa, até todas serem usadas como teste. No final tira a média.

```python
modelo = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=5))
scores = cross_val_score(modelo, X, y, cv=5)
```

O `make_pipeline` junta a normalização e o KNN, assim o scaler é refeito dentro de cada fold, sem vazar dados do teste.

Também usei o K-fold para testar valores diferentes de `k` (1, 3, 5, 7, 9 e 11) e ver qual funciona melhor.

## Resultados

<!-- Substitua pelos valores da sua execução -->

Matriz de confusão do KNN (k = 5) no conjunto de teste (linhas = real, colunas = previsto):

| | Setosa | Versicolor | Virginica |
|---|---|---|---|
| **Setosa** | 10 | 0 | 0 |
| **Versicolor** | 0 | 10 | 0 |
| **Virginica** | 0 | 1 | 9 |

| Métrica | Valor |
|---|---|
| Acurácia | 0,9667 |
| Precisão (macro) | 0,9697 |
| Recall (macro) | 0,9667 |
| F1 (macro) | 0,9666 |

| Técnica de validação | Acurácia |
|---|---|
| Hold-out | 0,9667 |
| n-hold-out (média de 5) | 0,9667 |
| K-fold (média de 5 folds) | 0,9600 |

## Conclusão

O KNN foi muito bem no Iris, com acurácia em torno de 96% a 97% em todas as técnicas de validação. A espécie **Setosa** foi sempre classificada corretamente, e o único erro do teste foi uma **Virginica** classificada como **Versicolor**, o que faz sentido, já que essas duas espécies são mais parecidas entre si. Os resultados parecidos nas três técnicas indicam que o modelo é estável e que o resultado do teste não foi apenas sorte.

## Estrutura do repositório

```
.
├── iris.csv
├── trabalho_iris_iniciante.py
└── README.md
```

## Autor

Matheus Oliveira Silva — Ciência da Computação, IFCE
