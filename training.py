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

def train_model(model, indices, ctxs, vocab_size, num_itterations, logspath, filespath):
    fitness = []
    batch_size = 10
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
            with open('trainingoutput/'+logspath, 'a') as f:
                f.write("\nConstructed first vector")

        x = np.array(x).reshape(len(x), vocab_size)
        y = np.array(y).reshape(len(y), vocab_size)
        y_hat, h = model.forward_step(x)

        if iter == 0:
            with open('trainingoutput//'+logspath, 'a') as f:
                f.write("\nFirst forward step")

        loss = y_hat - y
        fitness.append(cross_entropy_loss(y, y_hat))
        # print("\t", cross_entropy_loss(y, y_hat))
        model.backward_step(loss, 0.05, x, h)

        if iter == 0:
            with open('trainingoutput/'+logspath, 'a') as f:
                f.write("\nFirst backward step")

        if iter % 100 == 0:
            with open('trainingoutput/'+logspath, 'a') as f:
                f.write("\nIteration - " + str(iter))
                f.write("\t" + str(fitness[-1]))
    print(fitness)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    ax.plot(fitness)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Generation vs Fitness")
    fig.savefig("trainingoutput/fitnessgraph_"+filespath+".png")

    # Store data
    np.save("trainingoutput/embeddings_matrix_"+filespath, model.embedding)
    np.save("trainingoutput/context_matrix_"+filespath, model.context)
    np.save("trainingoutput/fitness_trajectory_"+filespath, fitness)

def training_setup(use_cooccurrence_matrix = False, seed_random = False):
    with open('trainingoutput/logs.txt', 'w') as f:
        f.write('Starting process')

    indices, ctxs, word_dict, vocab_size, u, s, vh = generate_training_data(3)

    with open('trainingoutput/word_dict_book7.pickle', 'wb') as handle:
        pickle.dump(word_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    num_itterations = 10000

    with open('trainingoutput/logs.txt', 'a') as f:
        f.write("\nFinished preprocessing, Vocab size = " + str(vocab_size))

    with open('trainingoutput/logs.txt', 'a') as f:
        f.write("\nInitialized model")

    if use_cooccurrence_matrix:
        model = skip_gram_model(u[:,:20], vh[:,:20].T)
    else:
        if seed_random:
            with open('trainingoutput/logs.txt', 'a') as f:
                f.write('Seeding random for matrices')
            np.random.seed(0)
        model = skip_gram_model(np.random.rand(vocab_size, 20), np.random.rand(20, vocab_size))

    if seed_random:
        with open('trainingoutput/logs.txt', 'a') as f:
            f.write('Seeding random')
        np.random.seed(0)

    with open('trainingoutput/logs.txt', 'a') as f:
        f.write("\nInitialized model")

    return model, indices, ctxs, vocab_size
