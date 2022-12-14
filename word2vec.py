import numpy as np
from numba import njit


# from scipy.special import softmax

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

    def forward_step(self, onehot):
        """
        Perform one forward step with the model, ie make a prediction of the context from a word

        :param onehot: A one hot encoded version of the matrix
        :return: a tuple containing: the output matrix
        """
        return forward(self.embedding, self.context, onehot)

    def backward_step(self, loss, learning_rate, onehot, h):
        """
        Perform one backpropagation step

        :param loss:
        :param learning_rate:
        :return:
        """
        d_dcontext, d_dembedding = backward(self.context, loss, onehot, h)
        self.context -= learning_rate * d_dcontext
        self.embedding -= learning_rate * d_dembedding

# @njit
def forward(embedding, context, onehot):
    h = onehot @ embedding
    i = h @ context
    # assert np.allclose(softmax(i, 1),custom_softmax(i, 1))
    return custom_softmax(i), h

# @njit
def backward(context, loss, onehot, h):
    d_dcontext = h.T @ loss
    d_dembedding = onehot.T @ loss @ context.T
    return d_dcontext, d_dembedding

@njit
def custom_softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    return np.divide(np.exp(x), np.sum(np.exp(x), axis).reshape(x.shape[0], 1))
