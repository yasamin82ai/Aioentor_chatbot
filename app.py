
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import answer_question_from_cached_embedding  # وارد کردن کد چت‌بات
import os
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی از فایل .env
load_dotenv()

app = Flask(__name__)
CORS(app)  # فعال‌سازی CORS برای تمامی درخواست‌ها

api_key = os.getenv("API_KEY")

# صفحه اصلی
@app.route('/')
def index():
    return render_template('index.html')

# دریافت پرسش از UI و ارسال پاسخ
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('message')
    pdf_path = "mymentor.pdf"  # مسیر فایل PDF شما
    try:
        response = answer_question_from_cached_embedding(pdf_path, question, api_key)
        return jsonify({"response": response})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
