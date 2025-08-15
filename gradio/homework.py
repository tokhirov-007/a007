import fasttext
import gradio as gr

# Путь к датасету
dataset_path = "dataset.txt"  # Файл должен лежать в одной папке со скриптом

# Обучение модели
print("Обучаем модель...")
model = fasttext.train_supervised(
    input=dataset_path,
    epoch=25,
    lr=1.0,
    wordNgrams=2,
    bucket=200000,
    dim=100,
    loss='softmax'
)
print("Обучение завершено.")

# Сохраняем модель
model_path = "lang_detect.ftz"
model.save_model(model_path)
print(f"Модель сохранена: {model_path}")

# Функция для предсказания языка
def predict_language(text):
    if not text.strip():
        return "Введите текст..."
    label, confidence = model.predict(text)
    lang = label[0].replace("__label__", "")
    conf = round(confidence[0] * 100, 2)
    return f"{lang} ({conf}%)"

# Gradio интерфейс с real-time
iface = gr.Interface(
    fn=predict_language,
    inputs=gr.Textbox(label="Введите текст"),
    outputs=gr.Textbox(label="Определённый язык"),
    title="🌍 Real-Time Language Detector",
    description="Мгновенное определение языка (uz, ru, en, kk, qq) во время ввода текста",
    live=True  # 🔥 Обновление при каждом вводе
)

# Запуск
if __name__ == "__main__":
    iface.launch()