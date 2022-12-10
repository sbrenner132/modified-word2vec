import numpy as np
from preprocess import generate_training_data
from word2vec import skip_gram_model
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_onehot(indices, vocab_size):
    if isinstance(indices, int):
        return create_single_onehot(indices, vocab_size)
    else:
        vector = np.zeros((1, vocab_size))
        for index in indices:
            vector += create_single_onehot(index, vocab_size)
        return vector

def create_single_onehot(index, vocab_size):
    vector = np.zeros((1, vocab_size))
    vector[0, index] = 1
    return vector

def cross_entropy_loss (y, y_hat):
  return - np.sum(y*np.log(y_hat))

indices, ctxs, _, vocab_size = generate_training_data(2)

num_itterations = 10000
print(vocab_size)
fitness = []
model = skip_gram_model(np.random.rand(vocab_size, 12), np.random.rand(12, vocab_size))
for iter in tqdm(range(num_itterations)):
    #print(iter)
    sample_number = np.random.randint(len(indices))
    x = create_onehot(indices[sample_number], vocab_size)
    y = create_onehot(ctxs[sample_number], vocab_size)
    y_hat, h = model.forward_step(x)
    loss = y_hat - y
    fitness.append(cross_entropy_loss(y, y_hat))
    #print("\t", cross_entropy_loss(y, y_hat))
    model.backward_step(loss, 0.1, x, h)
    if iter % 1000 == 0:
        print(loss)
print(fitness)
plt.plot(fitness)
