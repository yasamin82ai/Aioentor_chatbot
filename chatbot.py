import nltk
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import fitz  # برای خواندن PDF
import json
import os
from nltk.tokenize import sent_tokenize
from collections import deque  # برای محدود کردن تاریخچه به 10 مورد

# تنظیمات API Key برای OpenAI
api_key = os.getenv("API_KEY")

# استخراج متن از فایل PDF
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("فایل PDF یافت نشد. لطفاً مسیر فایل را بررسی کنید.")
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text("text")
    return text

# تقسیم متن به بخش‌های کوچک‌تر با حفظ جملات
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=400):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ذخیره و بارگذاری متن‌های تقسیم‌شده
def save_chunks_to_json(chunks, json_filename):
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

def load_chunks_from_json(json_filename):
    if not os.path.exists(json_filename):
        raise FileNotFoundError("فایل JSON وجود ندارد. لطفاً ابتدا فایل PDF پردازش شود.")
    with open(json_filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# ساخت embedding و FAISS
def create_embedding(chunks, api_key):
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_texts(chunks, embedding)  # حذف پارامتر index_type

# ذخیره FAISS Index
def save_faiss_index(vector_store, file_name="faiss_index"):
    vector_store.save_local(file_name)

# بارگذاری FAISS Index
def load_faiss_index(file_name="faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(file_name, embeddings, allow_dangerous_deserialization=True)

# تاریخچه مکالمه به صورت global یا در متغیر سطح بالا
chat_history = deque(maxlen=10)

prompt_template = """
You are "AioMentor" ("آیومنتور"). your name is , a friendly, warm, and intelligent assistant that responds in the user's preferred language (default is Persian). 
Your replies should always be concise, clear, and engaging. Ensure to use a good amount of spacing between sentences and points (line breaks) to make responses more readable.
You should use a friendly and casual tone, with a touch of enthusiasm! 
When responding, use bullet points or structured formatting where applicable, especially when listing features or steps. 
Use emojis naturally to enhance engagement and express emotions, but don't overuse them. Keep it light and fun! 
If the user asks a question, always try to give an answer that is not only informative but also encourages further interaction. 
Keep it fun and interactive! Encourage the user to ask more questions or share thoughts. Respond like you're having a casual chat with a friend, and always add a little sparkle to your answers! ✨🤩

Remember: Your goal is to make every conversation with the user feel friendly, warm, and exciting!
"""

# پاسخ به سوال از فایل PDF
def answer_question_from_cached_embedding(pdf_path, question, api_key, json_filename="pdf_chunks.json", index_name="faiss_index"):
    # اگر FAISS Index موجود است، از آن استفاده کن
    if os.path.exists(index_name):
        vector_store = load_faiss_index(index_name)
    else:
        # اگر فایل JSON موجود است، از آن استفاده کن
        if os.path.exists(json_filename):
            chunks = load_chunks_from_json(json_filename)
        else:
            # استخراج و تقسیم‌بندی متن PDF
            text = extract_text_from_pdf(pdf_path)
            chunks = split_text_into_chunks(text)
            save_chunks_to_json(chunks, json_filename)

        # ساخت FAISS Index و ذخیره‌سازی آن
        vector_store = create_embedding(chunks, api_key)
        save_faiss_index(vector_store, index_name)

    # ساخت Retriever برای جستجو در FAISS
    retriever = vector_store.as_retriever(search_type="similarity", search_k=1)  # کاهش تعداد جستجوها

    # تنظیمات مدل OpenAI
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o", temperature=0.7)  # استفاده از gpt-4

    # ساخت ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)

    # پرسش با پرامپت
    question_with_prompt = f"{prompt_template}{question}"

    # دریافت پاسخ و به‌روز کردن تاریخچه
    response = chain.invoke({"question": question_with_prompt, "chat_history": list(chat_history)})

    # اضافه کردن پرسش و پاسخ به تاریخچه برای حفظ مکالمه
    chat_history.append((question, response["answer"]))

    return response["answer"]
