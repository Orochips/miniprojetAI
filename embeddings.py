# importations nécessaires
import deeplake
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import data_loader

from langchain.vectorstores import DeepLake
from langchain.schema import Document
import config

# Configuration des embeddings
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Charger le dataset MedQA depuis Activeloop
medqa_dataset_path = "hub://noamaneoel/medqa-dataset"  # Chemin du dataset MedQA
ds = deeplake.load(medqa_dataset_path)

print(f"Dataset chargé: {len(ds)} exemples")
print(f"Colonnes disponibles: {ds.tensors.keys()}")

# Convertir les données MedQA en documents LangChain
documents = []

for i in range(len(ds)):
    # Extraire les champs (ajustez selon la structure de votre dataset)
    question = ds.question[i].text()
    
    # Construire le contenu du document
    # Option 1: Juste la question
    content = f"Question: {question}"
    
    # Option 2: Question + options (si disponibles)
    # options = ds.options[i].text()
    # content = f"Question: {question}\nOptions: {options}"
    
    # Option 3: Question + options + answer
    # answer = ds.answer[i].text()
    # content = f"Question: {question}\nOptions: {options}\nAnswer: {answer}"
    
    # Créer un document LangChain avec métadonnées
    doc = Document(
        page_content=content,
        metadata={
            "source": "medqa",
            "index": i,
            # "answer": answer,  # si vous voulez garder la réponse en métadonnée
        }
    )
    documents.append(doc)

print(f"Nombre de documents créés: {len(documents)}")

# Découpage si nécessaire (optionnel pour MedQA car les questions sont courtes)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)

print(f"Nombre de chunks après split: {len(texts)}")

# Créer le vector store dans un nouveau dataset Activeloop
vector_dataset_path = "hub://noamaneoel/medqa-vectorized"

db = DeepLake.from_documents(
    texts,
    embeddings,
    dataset_path=vector_dataset_path,
    token=config.token 
)

print("Vectorisation terminée et stockée dans Deep Lake!")
mayo_docs_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
mayo_texts=mayo_docs_splitter.split_documents(data_loader.mayo_docs)
vector_ds_path="hub://noamaneoel/mayo-vectorized"
db=DeepLake.from_documents(mayo_texts,embeddings,dataset_path=vector_ds_path,token=config.token)
print("Vectorisation des documents Mayo terminée et stockée dans Deep Lake!")
"""from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import DeepLake

from langchain_text_splitter import CharacterTextSplitter

from langchain_openai import OpenAI

from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import SeleniumURLLoader

from langchain_community.embeddings import HuggingFaceEmbeddings

import config

articles = ['https://www.digitaltrends.com/computing/claude-sonnet-vs-gpt-4o-comparison/',
           'https://www.digitaltrends.com/computing/apple-intelligence-proves-that-macbooks-need-something-more/',
           'https://www.digitaltrends.com/computing/how-to-use-openai-chatgpt-text-generation-chatbot/',
           'https://www.digitaltrends.com/computing/character-ai-how-to-use/',
           'https://www.digitaltrends.com/computing/how-to-upload-pdf-to-chatgpt/']

# Use the selenium to load the documents

docs_not_splitted = loader.load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)


# 3. Créer les embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "noamaneoel"
my_activeloop_dataset_name = "aiDataset"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings,token=config.token)


# add documents to our Deep Lake dataset
db.add_documents(docs)"""