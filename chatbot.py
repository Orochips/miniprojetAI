import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_community.llms import Ollama
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.memory import ConversationBufferMemory
from langchain_community.prompts import PromptTemplate

class MedicalChatbot:
    def __init__(self, vector_db_path, activeloop_token=None):
        """
        Initialise le chatbot médical
        
        Args:
            vector_db_path: Chemin vers la base vectorielle Deep Lake
            activeloop_token: Token Activeloop (optionnel si déjà dans env)
        """
        print("🔧 Initialisation du chatbot médical...")
        
        # Configuration des embeddings
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="NeuML/pubmedbert-base-embeddings"
        )
        
        # Charger la base vectorielle
        print("📚 Chargement de la base vectorielle...")
        self.db = DeepLake(
            dataset_path=vector_db_path,
            embedding_function=self.embeddings,
            token=activeloop_token or os.environ.get("ACTIVELOOP_TOKEN"),
            read_only=True
        )
        
        # Configuration du retriever
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Récupère les 5 documents les plus similaires
        )
        
        # Configuration du LLM (vous pouvez utiliser Ollama, OpenAI, ou autre)
        # Option 1: Ollama local (gratuit)
        self.llm = Ollama(model="llama2")  # ou "mistral", "medllama2", etc.
        
        # Option 2: OpenAI (nécessite API key)
        # from langchain_openai import ChatOpenAI
        # self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        
        # Mémoire de conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Prompt template personnalisé pour le contexte médical
        self.prompt_template = """You are a medical AI assistant specialized in answering medical questions based on the MedQA dataset.

Use the following context from medical knowledge to answer the question. If you don't know the answer based on the context, say so clearly.

Context from medical knowledge base:
{context}

Chat History:
{chat_history}

Question: {question}

Provide a clear, accurate, and professional medical answer. If relevant, mention the source or explain your reasoning:

Answer:"""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Créer la chaîne conversationnelle
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            verbose=False
        )
        
        print("✅ Chatbot prêt!\n")
    
    def ask(self, question):
        """
        Pose une question au chatbot
        
        Args:
            question: La question médicale
            
        Returns:
            dict: Réponse avec answer et source_documents
        """
        try:
            response = self.qa_chain({"question": question})
            return response
        except Exception as e:
            return {
                "answer": f"Erreur lors du traitement: {str(e)}",
                "source_documents": []
            }
    
    def chat(self):
        """
        Interface de chat interactive en ligne de commande
        """
        print("=" * 60)
        print("🏥 CHATBOT MÉDICAL - MedQA Assistant")
        print("=" * 60)
        print("\nTapez 'quit', 'exit' ou 'q' pour quitter")
        print("Tapez 'clear' pour effacer l'historique\n")
        
        while True:
            try:
                # Obtenir la question de l'utilisateur
                user_input = input("\n🧑 Vous: ").strip()
                
                # Commandes spéciales
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Au revoir!")
                    break
                
                if user_input.lower() == 'clear':
                    self.memory.clear()
                    print("🧹 Historique effacé!")
                    continue
                
                if not user_input:
                    continue
                
                # Obtenir la réponse
                print("\n🤖 Assistant: ", end="", flush=True)
                response = self.ask(user_input)
                print(response["answer"])
                
                # Afficher les sources si demandé
                if response.get("source_documents"):
                    show_sources = input("\n📎 Voir les sources? (o/n): ").lower()
                    if show_sources == 'o':
                        print("\n📚 Sources:")
                        for i, doc in enumerate(response["source_documents"][:3], 1):
                            print(f"\n--- Source {i} ---")
                            print(doc.page_content[:300] + "...")
                            if doc.metadata:
                                print(f"Métadonnées: {doc.metadata}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {str(e)}")
    
    def get_relevant_context(self, question, k=3):
        """
        Récupère le contexte pertinent sans générer de réponse
        
        Args:
            question: La question
            k: Nombre de documents à récupérer
            
        Returns:
            list: Liste de documents pertinents
        """
        docs = self.db.similarity_search(question, k=k)
        return docs


# ==================== UTILISATION ====================

if __name__ == "__main__":
    # Configuration
    VECTOR_DB_PATH = "hub://votre-username/medqa-vectorized"
    ACTIVELOOP_TOKEN = "votre-token"  # ou None si dans les variables d'env
    
    # Créer le chatbot
    chatbot = MedicalChatbot(
        vector_db_path=VECTOR_DB_PATH,
        activeloop_token=ACTIVELOOP_TOKEN
    )
    
    # Option 1: Mode interactif
    chatbot.chat()
    
    # Option 2: Questions individuelles
    # response = chatbot.ask("What are the symptoms of myocardial infarction?")
    # print(response["answer"])
    
    # Option 3: Récupérer seulement le contexte
    # docs = chatbot.get_relevant_context("What is diabetes?", k=3)
    # for doc in docs:
    #     print(doc.page_content)