import itertools
import string

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
    word_to_index, index = {}, 0  # start indexing from 1
    for word in words:
      if word not in word_to_index:
        word_to_index[word] = index
        index += 1
    return word_to_index

def generate_training_data(window_size):
    words = []
    for i in range(1):
      words += process_file(f'./corpus/Book{i + 1}.txt')
    words = [word for word in words if word not in ['the', 'to', 'of', 'a', 'and', 'in', 'that', 'have', 'i', 'be']]
    word_to_index = generate_word_to_index(words)
    
    indices, ctxs = [], []
    
    n = len(words)
    
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
      ctxs.append(word_ctx)
        
    return indices, ctxs, word_to_index, len(word_to_index.keys())
    
def concat(*iterables):
  for iterable in iterables:
    yield from iterable