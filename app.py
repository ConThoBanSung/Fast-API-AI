import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.schema import HumanMessage


os.environ["SERPAPI_API_KEY"] = "a88ba28bae03c781181e1df39d9722725b68020cd3b9c237f28d93f631db4f18"

# Thiết lập API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDrAggk6TX_A7Cp72Yb7VvJt694cKj6zxY"

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Khởi tạo mô hình Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Bộ nhớ hội thoại
memory = ConversationBufferMemory(memory_key="chat_history")

# Tạo Prompt
template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Lịch sử hội thoại:\n{chat_history}\n\nNgười dùng: {question}\nTrợ lý:"
)

# Chuỗi xử lý hội thoại
chat_chain = LLMChain(llm=llm, prompt=template, memory=memory)

# Công cụ tìm kiếm Google
search = SerpAPIWrapper()
tools = [
    Tool(name="Google Search", func=search.run, description="Tìm kiếm trên Google.")
]

# Tạo Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# API Model
class ChatRequest(BaseModel):
    message: str
    use_google: bool = False  # Mặc định không dùng Google Search

# API Chatbot
@app.post("/chat/")
async def chat(request: ChatRequest):
    if request.use_google:
        response = agent.invoke(request.message)
    else:
        response = chat_chain.invoke({"question": request.message})["text"]
    return {"response": response}

# Chạy server bằng lệnh: uvicorn app:app --reload
