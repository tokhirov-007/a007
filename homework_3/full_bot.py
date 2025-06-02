# full_bot.py

import pickle
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, ConversationHandler

# === Загрузка моделей ===
with open("models/price_by_rating.pkl", "rb") as f:
    model_reg_rating = pickle.load(f)

with open("models/price_by_text.pkl", "rb") as f:
    model_reg_text = pickle.load(f)

with open("models/category_classifier.pkl", "rb") as f:
    vector_cat, model_cat = pickle.load(f)

with open("models/price_class_classifier.pkl", "rb") as f:
    vector_price_class, model_price_class = pickle.load(f)

# === Состояния ===
RATING, TEXT, NAME, DESC = range(4)

# === Команды ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я ML-бот. Доступные команды:\n"
        "/predict_price_by_rating — Предсказать цену по рейтингу\n"
        "/predict_price_by_text — Предсказать цену по длине текста\n"
        "/classify_category — Определить категорию по названию\n"
        "/classify_price_class — Определить дорого/дёшево по описанию"
    )

# --- ЗАДАЧА 1: Регрессия по рейтингу ---
async def predict_price_by_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Введите рейтинг товара (например, 4.5):")
    return RATING

async def handle_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        rating = float(update.message.text)
        price = model_reg_rating.predict([[rating]])[0]
        await update.message.reply_text(f"💰 Предполагаемая цена: ${price:.2f}")
    except Exception:
        await update.message.reply_text("Ошибка! Введите число, например 3.8.")
    return ConversationHandler.END

# --- ЗАДАЧА 2: Регрессия по длине текста ---
async def predict_price_by_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Введите название и описание товара:")
    return TEXT

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    length = len(text)
    price = model_reg_text.predict([[length]])[0]
    await update.message.reply_text(f"💰 Прогнозируемая цена: ${price:.2f}")
    return ConversationHandler.END

# --- ЗАДАЧА 3: Классификация по названию ---
async def classify_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Введите название товара:")
    return NAME

async def handle_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.message.text
    X = vector_cat.transform([name])
    pred = model_cat.predict(X)[0]
    await update.message.reply_text(f"📦 Категория товара: {pred}")
    return ConversationHandler.END

# --- ЗАДАЧА 4: Классификация по описанию ---
async def classify_price_class(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Введите описание товара:")
    return DESC

async def handle_desc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    desc = update.message.text
    X = vector_price_class.transform([desc])
    pred = model_price_class.predict(X)[0]
    await update.message.reply_text(f"💲 Класс товара: {pred}")
    return ConversationHandler.END

# === Основной запуск ===
def main():
    app = ApplicationBuilder().token("8052695486:AAG_2gI7dftxfJnxKhLHyP0t488uYzxPnZI").build()

    # Команды
    app.add_handler(CommandHandler("start", start))

    # Задача 1
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("predict_price_by_rating", predict_price_by_rating)],
        states={RATING: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_rating)]},
        fallbacks=[],
    ))

    # Задача 2
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("predict_price_by_text", predict_price_by_text)],
        states={TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)]},
        fallbacks=[],
    ))

    # Задача 3
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("classify_category", classify_category)],
        states={NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_name)]},
        fallbacks=[],
    ))

    # Задача 4
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("classify_price_class", classify_price_class)],
        states={DESC: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_desc)]},
        fallbacks=[],
    ))

    print("🤖 Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
