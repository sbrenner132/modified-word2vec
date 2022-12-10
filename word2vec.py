import numpy as np
from scipy.special import softmax


class skip_gram_model():
    """
    A skip gram model to be trained with Batch Gradient DescentD
    """

    def __init__(self, initial_embedding, initial_context):
        """
        Initializing the skip gram model

        :param initial_embedding: initial weight matrix for hidden layer (W in Ken's in-class notation)
        :param initial_context: initial weight matrix for output layer (C in Ken's in-class notation)
        """
        self.embedding = initial_embedding
        self.context = initial_context
        self.last_step = {}

    def forward_step(self, onehot):
        """
        Perform one forward step with the model, ie make a prediction of the context from a word

        :param onehot: A one hot encoded version of the matrix
        :return: a tuple containing: the output matrix
        """
        self.last_step["onehot"] = onehot
        h = onehot @ self.embedding
        self.last_step["h"] = h
        i = h @ self.context
        x_hat = softmax(i)
        return x_hat

    def backward_step(self, loss, learning_rate):
        """
        Perform one backpropagation step

        :param loss:
        :param learning_rate:
        :return:
        """
        d_dcontext = self.last_step["h"].T @ loss
        d_dembedding = self.last_step["onehot"] @ loss @ self.context.T
        self.context -= learning_rate * d_dcontext
        self.embedding -= learning_rate * d_dembedding





