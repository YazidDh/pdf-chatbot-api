from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

# Multi language chatbot
from langchain.embeddings import HuggingFaceEmbeddings

from translator import translate_to_french


def load_db(file, chain_type, k):
    print("loading pdf ...")
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "score_threshold": 0.75}
    )
    """retriever = db.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": k, "neighbors": 3, "iterations": 2}
    )"""
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=OllamaLLM(model="qwen2.5:1.5b"),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )

    return qa



from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
from typing import List, Tuple


from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Add ALL dev origins if needed
origins = [
    "http://localhost:4200",  # Angular app
    "http://127.0.0.1:4200",  # Just in case
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] temporarily
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load QA system
start_load = time.time()
qa = load_db("rh_pdf/Rh_politiques.pdf", "stuff", 4)
print("loading time = ", time.time() - start_load)

# In-memory chat history
chat_history: List[Tuple[str, str]] = []

class Question(BaseModel):
    question: str

@app.post("/ask_RH_Chatbot")
def ask_question(q: Question):

    question = q.question
    res_time = time.time()

    result = qa({"question": question, "chat_history": chat_history})
    answer = translate_to_french(result["answer"])

    # Append to history
    chat_history.append((question, answer))

    return {
        "answer": answer,
        "response_time": time.time() - res_time
    }


