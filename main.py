from domain.entities import NeuralNetwork

import numpy as np
from tqdm import tqdm

from domain.fixture import num_epochs

if __name__ == '__main__':

    """
    Cria uma rede neural com 2 neurônios na camada de entrada, 3 neurônios na camada oculta e 1 neurônio na camada de saída
    """
    rede = NeuralNetwork(
        input_dim=2,
        hidden_dim=3,
        output_dim=1
    )

    """
        Define o conjunto de dados de entrada e saída esperada
    """
    entrada = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    esperado = np.array([[0], [1], [1], [0]])

    """
        Treina a rede neural por 1000 épocas
    """
    for i in tqdm(range(num_epochs)):
        rede.forward(entrada)

        # Calcula o erro
        error = esperado - rede.output_layer

        """
            Calcula o gradiente da camada de saída usando o erro e a derivada da função de ativação
        """
        output_gradient = error * rede.relu(rede.output_layer)

        """
            Calcula o gradiente da camada oculta usando o gradiente da camada de saída 
            e a derivada da função de ativação
        """
        hidden_gradient = np.dot(output_gradient, rede.weights2.T) * rede.relu(rede.hidden_layer)

        """
            Ajusta os pesos da rede
        """
        rede.weights2 += np.dot(rede.hidden_layer.T, output_gradient)
        rede.weights1 += np.dot(entrada.T, hidden_gradient)

    # Execução do teste
    test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    rede.forward(test)
    print(rede.output_layer)
