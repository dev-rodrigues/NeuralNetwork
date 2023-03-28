from domain.entities import NeuralNetwork
from domain.fixture import num_epochs

import numpy as np

if __name__ == '__main__':
    """
        Define o conjunto de dados de entrada e saída esperada
    """
    entrada = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    esperado = np.array([[0], [1], [1], [0]])

    """
        Cria uma rede neural:
        2 neurônios na camada de entrada 
        3 neurônios na camada oculta 
        1 neurônio na camada de saída
    """
    rede = NeuralNetwork(
        entrada=2,
        oculta=3,
        saida=1,
    )

    rede.train(
        rede=rede,
        entrada=entrada,
        esperado=esperado,
    )

    # Execução do teste
    validar = [[0, 0], [0, 1], [1, 0], [1, 1]]
    print(f"Dados de entrada {validar}")
    test = np.array(validar)
    rede.forward(test)

    print("Dados de output  [", ", ".join(str("[") + str(x[0]) + str("]") for x in rede.output_layer), sep="")

    print()

    print(f"Epochs {num_epochs}")

    # Calcular a acuracia
    predicoes = np.round(rede.output_layer)
    acuracia = np.mean(predicoes == esperado)

    print(f"Acurácia: {acuracia * 100}%")
