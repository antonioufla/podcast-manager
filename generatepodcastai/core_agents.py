from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub

class DocumentProcessingAgent:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vectorstore = None

    def process_documents(self, texts: List[str]):
        chunks = []
        for text in texts:
            chunks.extend(self.text_splitter.split_text(text))
        embeddings = self.embedding_model.embed_documents(chunks)
        self.vectorstore = Chroma.from_texts(chunks, self.embedding_model)
        return self.vectorstore

class QAAgent:
    def __init__(self, llm_model_name: str = "google/flan-t5-base"):
        self.llm = HuggingFaceHub(repo_id=llm_model_name)

    def answer_question(self, question: str, vectorstore):
        docs = vectorstore.similarity_search(question, k=3)
        context = " ".join([doc.page_content for doc in docs])
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response = self.llm(prompt)
        return response

class SummarizationAgent:
    def __init__(self, llm_model_name: str = "google/flan-t5-base"):
        self.llm = HuggingFaceHub(repo_id=llm_model_name)

    def summarize(self, text: str):
        prompt = f"Summarize the following text:\n{text}"
        summary = self.llm(prompt)
        return summary

class IdeationAgent:
    def __init__(self, llm_model_name: str = "google/flan-t5-base"):
        self.llm = HuggingFaceHub(repo_id=llm_model_name)

    def generate_ideas(self, prompt: str):
        response = self.llm(prompt)
        return response

class TopicExtractionAgent:
    def __init__(self, llm_model_name: str = "google/flan-t5-base"):
        self.llm = HuggingFaceHub(repo_id=llm_model_name)

    def extract_topics(self, text: str):
        prompt = f"Extract key topics and keywords from the following text:\n{text}"
        topics = self.llm(prompt)
        return topics
