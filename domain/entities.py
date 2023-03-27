import numpy as np


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.output_layer = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

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
