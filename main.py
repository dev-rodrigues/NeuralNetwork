from domain.entities import NeuralNetwork

import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    num_epochs = 100

    """
        Cria uma rede neural com 2 neurônios na camada de entrada, 3 neurônios na camada oculta e 1 neurônio na camada 
        de saída
    """
    nn = NeuralNetwork(2, 3, 1)

    """
        Define o conjunto de dados de entrada e saída esperada
    """
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    """
        Treina a rede neural por 1000 épocas
    """
    for i in tqdm(range(num_epochs)):
        nn.forward(x)

        # Calcula o erro
        error = y - nn.output_layer

        """
            Calcula o gradiente da camada de saída usando o erro e a derivada da função de ativação
        """
        output_gradient = error * nn.relu(nn.output_layer)

        """
            Calcula o gradiente da camada oculta usando o gradiente da camada de saída 
            e a derivada da função de ativação
        """
        hidden_gradient = np.dot(output_gradient, nn.weights2.T) * nn.relu(nn.hidden_layer)

        # Ajusta os pesos da rede
        nn.weights2 += np.dot(nn.hidden_layer.T, output_gradient)
        nn.weights1 += np.dot(x.T, hidden_gradient)

    # Execução do teste
    test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    nn.forward(test)
    print(nn.output_layer)
