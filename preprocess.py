import itertools
import string
import numpy as np
import matplotlib.pyplot as plt
import os.path

start_quote = '“'
end_quote = '”'

def filter_lines(file: str):
  data = open(file)
  filtered = []
  for line in data.readlines():
    line = line.strip()
    if line and not line.startswith('Page') and not line.isnumeric(): # ignore blank lines and page or chapter deliniations
      filtered.append(line) 
  return filtered

def process_line(line: str):
  words = line.split()
  return [word.strip(start_quote).strip(end_quote).translate(str.maketrans('', '', string.punctuation)).lower() for word in words]

def process_file(file: str):
  lines = filter_lines(file)
  words = [process_line(line) for line in lines]
  return list(itertools.chain(*words))

def generate_word_to_index(words):
    word_to_index, index = {}, 0
    for word in words:
      if word not in word_to_index:
        word_to_index[word] = index
        index += 1
    return word_to_index

def generate_training_data(window_size):
    with open('testrun/logs.txt', 'a') as f:
      f.write("\nBeginning pre-preprocessing")

    words =  process_file('./corpus/Book7.txt')
    words = [word for word in words if word not in ['the', 'to', 'of', 'a', 'and', 'in', 'that', 'have', 'i', 'be']]
    word_to_index = generate_word_to_index(words)
    
    indices, ctxs = [], []
    vocab_size = len(word_to_index.keys())
    
    n = len(words)
    with open('testrun/logs.txt', 'a') as f:
      f.write("\nMiddle of pre-preprocessing: word count = " + str(len(words)))

    cooccurrence = np.zeros(shape=(vocab_size, vocab_size)) # creating the co-occurence matrix

    for i in range(n):
      indices.append(word_to_index[words[i]])
      idx = concat(
        range(max(0, i - window_size), i), 
        range(i, min(n, i + window_size + 1))
      )


      word_ctx = []
      for j in idx:
        if i == j:
          continue
        word_ctx.append(word_to_index[words[j]])
        cooccurrence[word_to_index[words[i]]][word_to_index[words[j]]] += 1
      ctxs.append(word_ctx)


    # see if data is cached
    if os.path.isfile('testrun/u.npy') and os.path.isfile('testrun/s.npy') and os.path.isfile('testrun/vh.npy'):
      u = np.load('testrun/u.npy')
      s = np.load('testrun/s.npy')
      vh = np.load('testrun/vh.npy')
    else:
      u, s, vh = np.linalg.svd(cooccurrence, hermitian=True)
      np.save('testrun/u', u)
      np.save('testrun/s', s)
      np.save('testrun/vh', vh)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    ax.plot(s[:100])
    fig.savefig("testrun/s.png")

    with open('testrun/logs.txt', 'a') as f:
      f.write("\nFinished pre-preprocessing")

    return indices, ctxs, word_to_index, vocab_size, u, s, vh
    
def concat(*iterables):
  for iterable in iterables:
    yield from iterable