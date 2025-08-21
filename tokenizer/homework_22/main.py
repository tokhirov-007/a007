import numpy as np

# 1. Ma'lumot yig'ish
text = "Men AI model yaratmoqdaman va u juda zo'r ishlaydi"
print("Original text:", text)

# 2. Tokenizer qilish (разбиваем по словам)
tokens = text.split()
print("\nTokens:", tokens)

# 3. Vocabulary (word2idx)
vocab = {word: idx for idx, word in enumerate(set(tokens))}
print("\nVocabulary (word2idx):", vocab)

# Преобразуем токены в индексы
token_ids = [vocab[word] for word in tokens]
print("\nToken IDs:", token_ids)

# 4. Token Embedding (каждое слово в вектор размерности d_model)
d_model = 8  # размер вектора
embedding_matrix = np.random.rand(len(vocab), d_model)  # случайные вектора
print("\nEmbedding Matrix shape:", embedding_matrix.shape)

token_embeddings = np.array([embedding_matrix[idx] for idx in token_ids])
print("\nToken Embeddings (shape={}):".format(token_embeddings.shape))
print(token_embeddings)

# 5. Positional Embedding (синусоидальная формула)
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / d_model)
    angle_rads = pos * angle_rates
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pe

pos_emb = positional_encoding(len(token_ids), d_model)
print("\nPositional Embeddings (shape={}):".format(pos_emb.shape))
print(pos_emb)

# 6. Итоговое представление
final_embeddings = token_embeddings + pos_emb
print("\nFinal Embeddings (Token + Positional) (shape={}):".format(final_embeddings.shape))
print(final_embeddings)
