from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
import os
import tempfile
import logging
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

# Cấu hình logging để ghi lại log cho quá trình xử lý
logging.basicConfig(level=logging.INFO)

# Cấu hình API Key của Gemini
GENAI_API_KEY = "AIzaSyDrAggk6TX_A7Cp72Yb7VvJt694cKj6zxY"

# Thiết lập API Key cho Google Generative AI
genai.configure(api_key=GENAI_API_KEY)

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Biến lưu trữ FAISS vectorstore
vectorstore = None

# Hàm khởi tạo FAISS vectorstore
# FAISS giúp lưu trữ và tìm kiếm văn bản dựa trên embeddings
def initialize_vectorstore():
    global vectorstore
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS(embedding_function=embeddings)

# Hàm xử lý tệp tin tải lên
async def process_uploaded_file(file: UploadFile):
    file_extension = file.filename.split(".")[-1].lower()  # Lấy phần mở rộng của tệp tin
    file_size = len(await file.read())  # Đọc tệp để lấy kích thước
    await file.seek(0)  # Đưa con trỏ tệp về đầu để có thể đọc lại

    # Giới hạn kích thước tệp tin là 10MB
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the 10MB limit.")
    
    # Lưu tệp tin tạm thời trên máy chủ
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(await file.read())
        temp_filepath = temp_file.name
    
    # Lựa chọn phương thức đọc tệp dựa trên định dạng
    if file_extension == "txt":
        loader = TextLoader(temp_filepath)
    elif file_extension == "pdf":
        loader = PyPDFLoader(temp_filepath)
    elif file_extension in ["doc", "docx"]:
        loader = Docx2txtLoader(temp_filepath)
    else:
        raise HTTPException(status_code=400, detail="File format not supported.")
    
    # Đọc nội dung tệp và chia nhỏ văn bản để xử lý tốt hơn
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Lưu văn bản vào FAISS vectorstore
    global vectorstore
    embeddings = HuggingFaceEmbeddings()
    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        vectorstore.add_documents(docs)
    
    return "File processed and stored successfully"

# API Endpoint để tải tệp tin lên
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    message = await process_uploaded_file(file)
    return {"message": message}

# API Endpoint để trò chuyện với AI dựa trên dữ liệu tải lên
@app.post("/chat/")
async def chat(query: str):
    try:
        global vectorstore
        if vectorstore is None:
            raise HTTPException(status_code=400, detail="No documents uploaded yet.")
        
        # Tìm kiếm các tài liệu có liên quan dựa trên FAISS
        results = vectorstore.similarity_search(query, k=3)
        context = "\n".join([r.page_content for r in results])

        # Gửi câu hỏi và ngữ cảnh cho mô hình Gemini để tạo phản hồi
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(f"Context: {context}\nQuestion: {query}")
        answer = response.text
    except Exception as e:
        logging.error(f"Chat API error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")
    
    return {"response": answer}

# Chạy server FastAPI
if __name__ == "__main__":
    initialize_vectorstore()  # Khởi tạo vectorstore trước khi chạy server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
