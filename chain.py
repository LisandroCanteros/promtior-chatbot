import bs4
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Load env
load_dotenv()

## Model
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    # other params...
)

# Load website
website_loader = WebBaseLoader(
    web_paths=("https://www.promtior.ai/", "https://www.promtior.ai/service",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("PAGES_CONTAINER")
        )
    ),
)
docs = website_loader.load()


# Load pdf presentation
file_path = ("AI Engineer.pdf")
pdf_loader = PyPDFLoader(file_path)
docs += pdf_loader.load()

# Split documents into similar size chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")
print("----------------------------------")

# Index documents 
embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(embeddings)
retriever = vector_store.as_retriever()
document_ids = vector_store.add_documents(documents=all_splits)


# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("human", """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    Use three sentences maximum and keep the answer concise.
    
    Question: {question} 
    Context: {context} 
    Answer:""")
])


# Chain
input = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
chain = ( input | prompt | llm ).with_types(input_type=str)


 
