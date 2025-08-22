import numpy as np

# 1. Читаем файл
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Пример текста из файла:\n", text[:200])  # первые 200 символов

# 2. Токенизация (по пробелам, можно усложнить)
tokens = text.split()
print("\nВсего токенов:", len(tokens))
print("Пример токенов:", tokens[:20])

# 3. Vocabulary (словарь слов → индекс)
vocab_list = sorted(set(tokens))                # список уникальных слов
vocab = {word: i for i, word in enumerate(vocab_list)} 
print("\nРазмер словаря:", len(vocab))

# Преобразуем токены в индексы
token_ids = [vocab[word] for word in tokens]
print("Token IDs (первые 20):", token_ids[:20])

# 4. Token Embedding
d_model = 8  # размер вектора
np.random.seed(42)
embedding_matrix = np.random.rand(len(vocab), d_model)

token_embeddings = np.array([embedding_matrix[idx] for idx in token_ids])
print("\nToken Embeddings shape:", token_embeddings.shape)

# 5. Positional Encoding (синусоидальная)
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
print("\nPositional Embeddings shape:", pos_emb.shape)

# 6. Итоговое представление
final_embeddings = token_embeddings + pos_emb
print("\nFinal Embeddings shape:", final_embeddings.shape)
