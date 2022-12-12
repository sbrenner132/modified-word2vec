import numpy as np
from preprocess import generate_training_data
from word2vec import skip_gram_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from numba import njit

@njit
def create_onehot(index, vocab_size):
    vector = np.zeros((1, vocab_size))
    vector[0, index] = 1
    return vector

@njit
def cross_entropy_loss(y, y_hat):
    return - np.sum((y * np.log(y_hat)))


if __name__ == "__main__":

    with open('testrun/logs.txt', 'w') as f:
        f.write('Starting process')

    indices, ctxs, word_dict, vocab_size = generate_training_data(3)

    with open('testrun/word_dict_books12.pickle', 'wb') as handle:
        pickle.dump(word_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    num_itterations = 10000

    with open('testrun/logs.txt', 'a') as f:
        f.write("\nBegin training model, Vocab size = " + str(vocab_size))

    fitness = []
    batch_size = 10
    model = skip_gram_model((np.random.rand(vocab_size, 20) * 2) - 1, (np.random.rand(20, vocab_size) * 2) - 1)

    with open('testrun/logs.txt', 'a') as f:
        f.write("\nInitialized model")

    for iter in tqdm(range(num_itterations)):
        sample_numbers = np.random.randint(vocab_size, size=(batch_size))
        x = []
        y = []
        for sample_number in sample_numbers:
            word = indices[sample_number]
            for context in ctxs[sample_number]:
                x.append(create_onehot(word, vocab_size))
                y.append(create_onehot(context, vocab_size))

        if iter == 0:
            with open('testrun/logs.txt', 'a') as f:
                f.write("\nConstructed first vector")

        x = np.array(x).reshape(len(x), vocab_size)
        y = np.array(y).reshape(len(y), vocab_size)
        y_hat, h = model.forward_step(x)

        if iter == 0:
            with open('testrun/logs.txt', 'a') as f:
                f.write("\nFirst forward step")

        loss = y_hat - y
        fitness.append(cross_entropy_loss(y, y_hat))
        # print("\t", cross_entropy_loss(y, y_hat))
        model.backward_step(loss, 0.05, x, h)

        if iter == 0:
            with open('testrun/logs.txt', 'a') as f:
                f.write("\nFirst backward step")

        if iter % 100 == 0:
            with open('testrun/logs.txt', 'a') as f:
                f.write("\nIteration - " + str(iter))
                f.write("\t" + str(fitness[-1]))
    print(fitness)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    ax.plot(fitness)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Generation vs Fitness")
    fig.savefig("testrun/fitnessgraph.png")

    # Store data
    np.save("testrun/embeddings_matrix_test_books12", model.embedding)
    np.save("testrun/context_matrix_test_books12", model.context)
