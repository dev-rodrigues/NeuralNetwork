import numpy as np
from tqdm import tqdm

from domain.fixture import num_epochs


class NeuralNetwork:
    def __init__(self, entrada, oculta, saida):
        self.output_layer = None
        self.input_dim = entrada
        self.hidden_dim = oculta
        self.output_dim = saida

        """
            Inicializa os pesos da rede
        """
        self.weights1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.weights2 = np.random.randn(self.hidden_dim, self.output_dim)

        """
            Define a camada oculta como um atributo de instância
        """
        self.hidden_layer = None

    def forward(self, x):
        """
            Calcula a ativação da camada oculta usando a função de ativação ReLU
        """
        self.hidden_layer = self.relu(np.dot(x, self.weights1))

        """
            Calcula a ativação da camada de saída usando a função de ativação sigmoide
        """
        output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights2))

        """
            Define a camada de saída como um atributo de instância
        """
        self.output_layer = output_layer
        return output_layer

    def relu(self, input):
        return np.maximum(0, input)

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def train(self, rede, entrada, esperado):
        for _ in tqdm(range(num_epochs)):
            rede.forward(entrada)

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

            """
                atualiza os pesos da camada de saída
                retorna uma matriz de pesos atualizados que é adicionada aos pesos antigos rede.weights2
                T = transposta, ou seja, 
                
                A = [[1, 2, 3],
                    [4, 5, 6]]
                    
                At = [[1, 4],
                        [2, 5],
                        [3, 6]]
            """
            rede.weights2 += np.dot(rede.hidden_layer.T, output_gradient)

            """
                Atualiza os pesos da camada oculta
            """
            rede.weights1 += np.dot(entrada.T, hidden_gradient)
