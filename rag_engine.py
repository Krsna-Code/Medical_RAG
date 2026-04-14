import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def build_rag_chain(pdf_path: str):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Free local embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Prompt
    prompt = PromptTemplate.from_template("""
You are a medical document assistant. 
Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't have enough information in this document."
Always end your answer with: "Source: {source}"

Context: {context}

Question: {question}

Answer:
""")

    def format_docs(docs):
        sources = set(doc.metadata.get("source", "document") for doc in docs)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context, "source": ", ".join(sources)}

    chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["docs"])["context"],
            source=lambda x: format_docs(x["docs"])["source"]
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
