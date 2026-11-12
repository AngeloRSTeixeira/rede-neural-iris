# Rede Neural do Zero — Classificação Iris

Rede Neural Artificial com uma camada escondida, implementada do zero em Python usando apenas **NumPy** para os cálculos. O Scikit-Learn é usado somente para carregar e preparar os dados.

Trabalho da disciplina de Inteligência Artificial — Ciência da Computação, UNIPAC Barbacena.

## Arquitetura

| Camada     | Neurônios | Ativação |
|------------|-----------|----------|
| Entrada    | 4         | —        |
| Escondida  | 8         | ReLU     |
| Saída      | 3         | Softmax  |

- **Função de perda:** Cross-Entropy
- **Otimização:** Gradiente Descendente com backpropagation (regra da cadeia)
- **Épocas:** 1000 | **Taxa de aprendizado:** 0.1

## Pré-processamento

- Padronização das features (média 0, desvio padrão 1)
- One-Hot Encoding dos rótulos
- Divisão 80% treino / 20% teste

## Resultado

**Acurácia no conjunto de teste: 96,67%**

## Como executar

    pip install -r requirements.txt
    python rede_neural_iris.py
