# parse_data.py

import requests
from bs4 import BeautifulSoup
import sqlite3

BASE_URL = "https://webscraper.io/test-sites/e-commerce/static"
CATEGORY_URLS = [
    ("laptop", f"{BASE_URL}/computers/laptops"),
    ("tablet", f"{BASE_URL}/computers/tablets"),
    ("phone", f"{BASE_URL}/phones/touch")
]

products = []

for category, url in CATEGORY_URLS:
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    for item in soup.select(".thumbnail"):
        name = item.select_one(".title").text.strip()
        price = item.select_one(".price").text.replace("$", "").strip()
        desc = item.select_one(".description").text.strip()

        # Рейтинг делаем искусственным — длина описания / 10
        rating = len(desc) / 10.0

        # Метка для дешёвый/дорогой (граница: $700)
        price_value = float(price)
        price_class = "cheap" if price_value < 700 else "expensive"

        products.append((name, desc, price_value, rating, category, price_class))

# Создание базы данных
conn = sqlite3.connect("products.db")
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS products")

cur.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        price REAL,
        rating REAL,
        category TEXT,
        price_class TEXT
    )
""")

cur.executemany("""
    INSERT INTO products (name, description, price, rating, category, price_class)
    VALUES (?, ?, ?, ?, ?, ?)
""", products)

conn.commit()
conn.close()

print("✅ Данные успешно собраны и сохранены в 'products.db'")
