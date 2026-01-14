from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import DeepLake

from langchain_text_splitter import CharacterTextSplitter

from langchain_openai import OpenAI

from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import SeleniumURLLoader

from langchain_community.embeddings import HuggingFaceEmbeddings


articles = ['https://www.digitaltrends.com/computing/claude-sonnet-vs-gpt-4o-comparison/',
           'https://www.digitaltrends.com/computing/apple-intelligence-proves-that-macbooks-need-something-more/',
           'https://www.digitaltrends.com/computing/how-to-use-openai-chatgpt-text-generation-chatbot/',
           'https://www.digitaltrends.com/computing/character-ai-how-to-use/',
           'https://www.digitaltrends.com/computing/how-to-upload-pdf-to-chatgpt/']

# Use the selenium to load the documents
loader = SeleniumURLLoader(urls=articles)
docs_not_splitted = loader.load()

# TODO : Split the documents into smaller chunks
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
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)


# add documents to our Deep Lake dataset
db.add_documents(docs)