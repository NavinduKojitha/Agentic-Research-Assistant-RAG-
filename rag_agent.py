from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import os

class RAGAgent:
    def __init__(self, pdf_path: str, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.loader = PyPDFLoader(pdf_path)
        self.docs = self.loader.load()
        self.embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(self.docs, self.embeddings)
        self.retriever = self.db.as_retriever()
        self.llm = ChatOpenAI(temperature=0.1)
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever)

    def query(self, question: str) -> str:
        result = self.qa_chain.run(question)
        return result
