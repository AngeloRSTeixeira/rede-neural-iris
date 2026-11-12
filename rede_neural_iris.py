"""
Atividade Aula 10 - Inteligência Artificial (UNIPAC Barbacena)
Aluno: Angelo Ricardo de Souza Teixeira
Rede Neural Artificial com uma camada escondida, implementada do zero com NumPy,
para classificar as espécies do conjunto de dados Iris.

Scikit-Learn é usado apenas para carregar e preparar os dados.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

np.random.seed(42)

# =============================================================================
# 1. PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================
iris = load_iris()
X = iris.data                    # (150, 4) -> 4 características
y = iris.target.reshape(-1, 1)   # (150, 1) -> 3 classes (0, 1, 2)

# Padronização: média 0 e desvio padrão 1
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-Hot Encoding: cada rótulo vira um vetor binário de tamanho 3
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)     # (150, 3)

# Divisão treino (80%) / teste (20%)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 2. ARQUITETURA DA REDE NEURAL
# =============================================================================
n_entrada = 4     # 4 características
n_escondida = 8   # 8 neurônios na camada escondida
n_saida = 3       # 3 classes


def relu(z):
    return np.maximum(0, z)


def relu_derivada(z):
    return (z > 0).astype(float)


def softmax(z):
    # Subtrai o máximo de cada linha para estabilidade numérica
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Inicialização aleatória dos pesos (He, adequada para ReLU) e vieses
W1 = np.random.randn(n_entrada, n_escondida) * np.sqrt(2.0 / n_entrada)
b1 = np.zeros((1, n_escondida))
W2 = np.random.randn(n_escondida, n_saida) * np.sqrt(2.0 / n_escondida)
b2 = np.zeros((1, n_saida))


def forward(X):
    Z1 = X @ W1 + b1     # combinação linear da camada escondida
    A1 = relu(Z1)        # ativação ReLU
    Z2 = A1 @ W2 + b2    # combinação linear da camada de saída
    A2 = softmax(Z2)     # probabilidades das 3 classes
    return Z1, A1, Z2, A2


def cross_entropy(Y, A2):
    eps = 1e-12  # evita log(0)
    return -np.mean(np.sum(Y * np.log(A2 + eps), axis=1))


def acuracia(Y, A2):
    return np.mean(np.argmax(A2, axis=1) == np.argmax(Y, axis=1))


# =============================================================================
# 3. TREINAMENTO VIA GRADIENTE DESCENDENTE
# =============================================================================
epocas = 1000
taxa_aprendizado = 0.1
m = X_train.shape[0]  # número de amostras de treino

for epoca in range(1, epocas + 1):
    # ----- Forward Pass -----
    Z1, A1, Z2, A2 = forward(X_train)

    # ----- Cálculo da Perda (Cross-Entropy) -----
    loss = cross_entropy(Y_train, A2)

    # ----- Backward Pass (Backpropagation / regra da cadeia) -----
    # Softmax + Cross-Entropy: dL/dZ2 = A2 - Y
    dZ2 = (A2 - Y_train) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivada(Z1)
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # ----- Atualização dos pesos (Gradiente Descendente) -----
    W1 -= taxa_aprendizado * dW1
    b1 -= taxa_aprendizado * db1
    W2 -= taxa_aprendizado * dW2
    b2 -= taxa_aprendizado * db2

    if epoca % 100 == 0 or epoca == 1:
        print(f"Época {epoca:4d} | Loss: {loss:.4f} | "
              f"Acurácia treino: {acuracia(Y_train, A2) * 100:.2f}%")

# =============================================================================
# 4. AVALIAÇÃO DO MODELO
# =============================================================================
_, _, _, A2_teste = forward(X_test)
acc_teste = acuracia(Y_test, A2_teste)

print("\n" + "=" * 50)
print(f"Acurácia final no conjunto de teste: {acc_teste * 100:.2f}%")
print("=" * 50)