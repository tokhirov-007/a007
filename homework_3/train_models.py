# train_models.py

import sqlite3
import os
import pickle
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

# Загружаем данные
conn = sqlite3.connect("products.db")
df = pd.read_sql("SELECT * FROM products", conn)
conn.close()

# --- MODEL 1: Regression — rating → price ---
X1 = df[["rating"]]
y1 = df["price"]

model1 = LinearRegression()
model1.fit(X1, y1)

with open("models/price_by_rating.pkl", "wb") as f:
    pickle.dump(model1, f)

print("✅ Модель 1 (rating → price) сохранена.")

# --- MODEL 2: Regression — text length → price ---
df["text_length"] = df["name"].apply(len) + df["description"].apply(len)
X2 = df[["text_length"]]
y2 = df["price"]

model2 = LinearRegression()
model2.fit(X2, y2)

with open("models/price_by_text.pkl", "wb") as f:
    pickle.dump(model2, f)

print("✅ Модель 2 (text length → price) сохранена.")

# --- MODEL 3: Classification — name → category ---
vectorizer3 = CountVectorizer()
X3 = vectorizer3.fit_transform(df["name"])
y3 = df["category"]

model3 = MultinomialNB()
model3.fit(X3, y3)

with open("models/category_classifier.pkl", "wb") as f:
    pickle.dump((vectorizer3, model3), f)

print("✅ Модель 3 (name → category) сохранена.")

# --- MODEL 4: Classification — description → cheap/expensive ---
vectorizer4 = CountVectorizer()
X4 = vectorizer4.fit_transform(df["description"])
y4 = df["price_class"]

model4 = MultinomialNB()
model4.fit(X4, y4)

with open("models/price_class_classifier.pkl", "wb") as f:
    pickle.dump((vectorizer4, model4), f)

print("✅ Модель 4 (description → price class) сохранена.")
