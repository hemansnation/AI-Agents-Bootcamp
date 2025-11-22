import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

loader = PyPDFLoader("resume.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embeddings, 
                                    persist_directory="./chroma_db")

print("Ingestion complete! Indexted.", len(splits), "chunks.")
