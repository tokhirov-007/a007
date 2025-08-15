import fasttext
import gradio as gr
import os
import sys

dataset_path = "dataset.txt"
if not os.path.exists(dataset_path):
    print(f"❌ Файл {dataset_path} не найден.")
    sys.exit(1)

print("📚 Начинаем обучение модели...")
epochs = 30
temp_model_path = "temp_model.ftz"

model = None
for epoch in range(1, epochs + 1):
    model = fasttext.train_supervised(
        input=dataset_path,
        epoch=1,  
        lr=1.0,
        wordNgrams=2,
        bucket=200000,
        dim=100,
        loss='softmax',
        verbose=0  
    )
    
    model.save_model(temp_model_path)
    
    percent = round((epoch / epochs) * 100, 2)
    print(f"🔄 Эпоха {epoch}/{epochs} — {percent}% завершено")

print("✅ Обучение завершено.")

model_path = "lang_detect.ftz"
model.save_model(model_path)
print(f"💾 Модель сохранена: {model_path}")

def predict_language(text):
    if not text.strip():
        return "Введите текст..."
    label, confidence = model.predict(text)
    lang = label[0].replace("__label__", "")
    conf = round(confidence[0] * 100, 2)
    return f"{lang} ({conf}%)"

iface = gr.Interface(
    fn=predict_language,
    inputs=gr.Textbox(label="Введите текст"),
    outputs=gr.Textbox(label="Определённый язык"),
    title="🌍 Real-Time Language Detector",
    description="Мгновенное определение языка (uz, ru, en, kk, qq) во время ввода текста",
    live=True
)
# Оценка точности на тестовом датасете
test_path = "dataset.txt"  # файл с тестовыми примерами в формате fastText
if os.path.exists(test_path):
    result = model.test(test_path)
    print(f"\n📊 Результаты на тесте:")
    print(f"  Кол-во примеров: {result[0]}")
    print(f"  Precision: {round(result[1]*100, 2)}%")
    print(f"  Recall:    {round(result[2]*100, 2)}%")
else:
    print("\n⚠️ test.txt не найден, пропускаем оценку точности.")


if __name__ == "__main__":
    iface.launch(share=True)
    
