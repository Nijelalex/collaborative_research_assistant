# src/rag_pipeline.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def build_vectorstore(data_dir="data/papers", persist_dir="embeddings/chroma"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []

    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, file))
            docs.extend(loader.load())

    texts = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def load_vectorstore(persist_dir="embeddings/chroma"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
