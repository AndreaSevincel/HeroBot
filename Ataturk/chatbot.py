import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import os
import chardet

def convert_to_utf8(folder):
    backup_folder = os.path.join(folder, "backup")
    os.makedirs(backup_folder, exist_ok=True)

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            backup_path = os.path.join(backup_folder, filename)

            # Backup original file
            os.rename(file_path, backup_path)

            # Detect file encoding
            with open(backup_path, "rb") as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result["encoding"]

            if encoding is None:
                print(f"Could not detect encoding for {filename}, skipping.")
                continue

            print(f"Converting {filename} from {encoding} to UTF-8...")

            # Convert to UTF-8
            with open(backup_path, "r", encoding=encoding, errors="ignore") as f:
                content = f.read()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    print("Conversion completed! Original files are in the 'backup' folder.")


convert_to_utf8("text")


def load_documents_from_folders(folder_paths):
    documents = []
    for folder in folder_paths:
        for file_name in os.listdir(folder):
            if file_name.endswith(".txt"):  # Only process text files
                file_path = os.path.join(folder, file_name)
                loader = TextLoader(file_path)
                documents.extend(loader.load())  # Append to document list
    return documents

folders = ["text"]
documents = load_documents_from_folders(folders)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key="YOUR_API_KEY")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./db")

retriever = vectorstore.as_retriever()

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Sen Ulu Önder Mustafa Kemal Atatürksün, Türkiye Cumhuriyetinin kurucusu. Sen sadece hayatını, ve 
hayattayken olanları biliyorsunç Eğer 1938'den sonra bir şey sorarsam sen bilemeyeceksın, çünkü Atatürk 10 kasım 1938de
öldü.
Ama eğer sana 'sen bilmiyorsun ama ...' gibi, sadece senin görüşlerini,
bakış açını bilmek istiyorsam sen kendi hayatından bakarak kendi görüşlerini
diyeceksin.
Konuştuğun kişiyle bir baba-oğul dinamiği yarat, ona oğlum, evladım diye
hitap et.
Lütfen  bilmediğin, emin olmadığın şeyleri söyleme.
Oldukça gerçekçi olmaya çalış, sanki gerçekten Atatürk ile konuşuyormuş gibi.
Context: {context}
Question: {question}
Answer:"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4", openai_api_key="YOUR_API_KEY"),
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)

print("Atatürk Chatbot: Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break
    answer = qa_chain.run(query)
    print(f"Atatürk: {answer}\n")
