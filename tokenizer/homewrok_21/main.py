# main.py
import os
import requests
import nltk
from nltk.tokenize import word_tokenize

# --- 1. Скачиваем ресурсы NLTK ---
nltk.download("punkt")
nltk.download("punkt_tab")  # чтобы не было ошибки

# --- 2. Скачиваем текстовый датасет ---
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
dataset_path = "dataset.txt"

if not os.path.exists(dataset_path):
    print("Скачиваю датасет...")
    response = requests.get(url)
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Датасет сохранён:", dataset_path)
else:
    print("Датасет уже существует:", dataset_path)

# --- 3. Читаем датасет ---
with open(dataset_path, "r", encoding="utf-8") as f:
    big_text = f.read()

print("Размер текста:", len(big_text), "символов")

# --- 4. Токенизация ---
tokens = word_tokenize(big_text)

print("Количество токенов:", len(tokens))
print("Пример токенов:", tokens[:30])

# --- 5. Частотный словарь ---
from collections import Counter
freq = Counter(tokens)

print("Топ-10 самых частых токенов:")
for word, count in freq.most_common(10):
    print(word, ":", count)
