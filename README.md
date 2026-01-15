# 🏥 Medical Chatbot with MedQA Dataset

An intelligent medical question-answering chatbot powered by the MedQA dataset, PubMedBERT embeddings, and LangChain. This project creates a vector database from medical questions and provides an AI assistant capable of answering medical queries with context-aware responses.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![Deep Lake](https://img.shields.io/badge/Deep%20Lake-Activeloop-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **🧠 Medical Knowledge Base**: Built on the MedQA USMLE dataset with thousands of medical questions
- **🔍 Semantic Search**: Uses PubMedBERT embeddings optimized for medical terminology
- **💬 Conversational AI**: Maintains context across multiple turns of conversation
- **📚 Source Citations**: Provides relevant source documents for each answer
- **🎨 Multiple Interfaces**: CLI, Flask web app, and Streamlit dashboard
- **💾 Persistent Storage**: Vector database stored in Activeloop Deep Lake
- **🔄 Memory Management**: Conversation history with clear/reset functionality
- **⚡ Fast Retrieval**: Optimized vector similarity search

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embeddings     │ PubMedBERT
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Search  │ Deep Lake
│  (Top K docs)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Chain     │ Llama2/GPT
│  + Context +    │
│  Chat History   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AI Response    │
└─────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Ollama (for local LLM) or OpenAI API key
- Activeloop account and token

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama (Optional, for local LLM)

```bash
# Visit https://ollama.ai/download
# Then pull the model
ollama pull llama2

# Or use a medical-specific model
ollama pull medllama2
```

### Step 5: Setup Activeloop

1. Create account at [activeloop.ai](https://www.activeloop.ai/)
2. Get your API token from the dashboard
3. Set environment variable (see Configuration)

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Activeloop Configuration (Required)
ACTIVELOOP_TOKEN=your-activeloop-token-here

# OpenAI Configuration (Optional, if using OpenAI)
OPENAI_API_KEY=your-openai-api-key-here

# Anthropic Configuration (Optional, if using Claude)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Vector Database Path
VECTOR_DB_PATH=hub://your-username/medqa-vectorized
```

### Configuration Options

Edit the configuration section in the scripts:

```python
# In chatbot.py or app files
VECTOR_DB_PATH = "hub://your-username/medqa-vectorized"
MODEL_NAME = "llama2"  # or "gpt-3.5-turbo", "claude-3-sonnet"
EMBEDDINGS_MODEL = "NeuML/pubmedbert-base-embeddings"
TOP_K_RESULTS = 5  # Number of relevant documents to retrieve
```

## 📊 Data Preparation

### Step 1: Create Vector Database from MedQA

```bash
python embeddings.py
```

This script will:
1. Load the MedQA dataset from Activeloop
2. Convert questions and answers to documents
3. Generate embeddings using PubMedBERT
4. Store vectors in Deep Lake

Expected output:
```
Dataset chargé: 12723 exemples
Colonnes disponibles: dict_keys(['question', 'options', 'answer', 'answer_idx'])
Nombre de documents créés: 12723
Vectorisation et stockage...
✓ Vectorisation terminée!
```

### Step 2: Verify Vector Database

```python
import deeplake
from langchain_community.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
db = deeplake.load("hub://your-username/medqa-vectorized")

print(f"Total vectors: {len(db)}")
```

## 💻 Usage

### 1. Command Line Interface (CLI)

```bash
python chatbot.py
```

**Interactive Session:**
```
🏥 CHATBOT MÉDICAL - MedQA Assistant
====================================

Tapez 'quit', 'exit' ou 'q' pour quitter
Tapez 'clear' pour effacer l'historique

🧑 Vous: What are the symptoms of diabetes?

🤖 Assistant: Diabetes typically presents with several key symptoms...

📎 Voir les sources? (o/n): o

📚 Sources:
--- Source 1 ---
Question: A 45-year-old patient presents with increased thirst...
```

**Commands:**
- `quit`, `exit`, `q` - Exit the chatbot
- `clear` - Clear conversation history
- Any question - Ask the chatbot

### 2. Flask Web Application

```bash
python chatbot_flask.py
```

Then open your browser to: `http://localhost:5000`

**Features:**
- Beautiful modern UI
- Real-time chat interface
- Session management
- Mobile responsive
- Source document viewing

**API Endpoints:**
- `GET /` - Web interface
- `POST /ask` - Ask a question
- `POST /clear` - Clear history
- `GET /health` - Health check

### 3. Streamlit Dashboard

```bash
streamlit run chatbot_streamlit.py
```

**Features:**
- Interactive sidebar controls
- Message history
- Expandable source citations
- Statistics dashboard
- Example questions
- Easy configuration

### 4. Programmatic Usage

```python
from chatbot import MedicalChatbot

# Initialize
chatbot = MedicalChatbot(
    vector_db_path="hub://your-username/medqa-vectorized",
    activeloop_token="your-token"
)

# Ask a question
response = chatbot.ask("What is the treatment for hypertension?")
print(response["answer"])

# View sources
for doc in response["source_documents"]:
    print(doc.page_content)

# Get relevant context only
docs = chatbot.get_relevant_context("diabetes symptoms", k=3)
```

## 📁 Project Structure

```
medical-chatbot/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── embeddings.py              # Vector database creation
├── chatbot.py                 # CLI chatbot implementation
├── chatbot_flask.py           # Flask web application
├── chatbot_streamlit.py       # Streamlit dashboard
│
├── data/                      # Optional: Additional PDF documents
│   └── medical_papers/
│
├── tests/
│   ├── test_embeddings.py
│   ├── test_chatbot.py
│   └── test_api.py
│
└── docs/
    ├── API.md
    ├── DEPLOYMENT.md
    └── EXAMPLES.md
```

## 🔌 API Documentation

### Flask API Endpoints

#### POST /ask
Ask a question to the chatbot.

**Request:**
```json
{
  "question": "What are the symptoms of diabetes?",
  "session_id": "unique-session-id"
}
```

**Response:**
```json
{
  "answer": "Diabetes typically presents with...",
  "sources": [
    "Question: A patient with diabetes...",
    "Question: Common symptoms include..."
  ]
}
```

#### POST /clear
Clear conversation history for a session.

**Request:**
```json
{
  "session_id": "unique-session-id"
}
```

**Response:**
```json
{
  "status": "cleared"
}
```

#### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy"
}
```

## 📚 Examples

### Example 1: Diagnostic Question

**Input:**
```
What are the key symptoms that distinguish bacterial pneumonia from viral pneumonia?
```

**Output:**
```
Bacterial pneumonia typically presents with more acute onset and higher fever compared 
to viral pneumonia. Key distinguishing features include:

1. Bacterial: Productive cough with purulent sputum, high fever (>39°C), pleuritic 
   chest pain, and elevated white blood cell count
2. Viral: Gradual onset, dry cough, lower-grade fever, myalgias, and normal or 
   slightly elevated WBC count

Chest X-ray findings also differ, with bacterial showing lobar consolidation and 
viral showing interstitial patterns.

[Sources: 3 relevant MedQA questions cited]
```

### Example 2: Treatment Question

**Input:**
```
What is the first-line treatment for uncomplicated hypertension in a 55-year-old patient?
```

**Output:**
```
For uncomplicated hypertension in a 55-year-old patient, first-line treatment options 
include:

1. ACE inhibitors (e.g., lisinopril) or ARBs (e.g., losartan)
2. Thiazide diuretics (e.g., hydrochlorothiazide)
3. Calcium channel blockers (e.g., amlodipine)

The choice depends on comorbidities and patient-specific factors. Lifestyle 
modifications (diet, exercise, weight loss) should always accompany pharmacotherapy.

[Sources: 2 relevant MedQA questions cited]
```

### Example 3: Differential Diagnosis

**Input:**
```
A patient presents with acute chest pain and shortness of breath. What are the most 
critical diagnoses to rule out?
```

**Output:**
```
The most critical diagnoses requiring immediate evaluation include:

1. **Acute Coronary Syndrome (ACS)**: Unstable angina or MI
   - ECG, cardiac enzymes, risk stratification
   
2. **Pulmonary Embolism (PE)**: 
   - D-dimer, CT angiography, Wells score
   
3. **Aortic Dissection**:
   - CXR, CT angiography, blood pressure differential
   
4. **Tension Pneumothorax**:
   - Physical exam, CXR if stable
   
5. **Cardiac Tamponade**:
   - Echocardiography, clinical signs

Initial management includes oxygen, IV access, monitoring, and immediate diagnostic 
workup based on clinical presentation.

[Sources: 5 relevant MedQA questions cited]
```

## 🐛 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'langchain.text_splitter'

**Solution:**
```bash
pip install langchain-text-splitters
pip install langchain-community
```

#### 2. Activeloop Authentication Error

**Solution:**
```bash
# Set your token
export ACTIVELOOP_TOKEN="your-token-here"

# Or add to .env file
echo "ACTIVELOOP_TOKEN=your-token-here" >> .env
```

#### 3. Ollama Connection Error

**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# Pull the model
ollama pull llama2

# Test the model
ollama run llama2
```

#### 4. Out of Memory Error

**Solution:**
- Reduce `chunk_size` in text splitter
- Reduce `k` (number of retrieved documents)
- Use a smaller embedding model
- Process data in batches

```python
# In embeddings.py
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Reduced from 1000
    chunk_overlap=50
)

# In chatbot.py
search_kwargs={"k": 3}  # Reduced from 5
```

#### 5. Slow Response Times

**Solution:**
- Use a faster LLM (e.g., gpt-3.5-turbo instead of gpt-4)
- Reduce number of retrieved documents
- Use GPU acceleration for embeddings
- Cache frequently asked questions

### Debug Mode

Enable verbose logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# In chatbot initialization
qa_chain = ConversationalRetrievalChain.from_llm(
    ...,
    verbose=True  # Enable detailed logging
)
```

## 🧪 Testing

Run the test suite:

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Run specific test file
pytest tests/test_chatbot.py -v
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ACTIVELOOP_TOKEN=${ACTIVELOOP_TOKEN}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

EXPOSE 5000

CMD ["python", "chatbot_flask.py"]
```

Build and run:
```bash
docker build -t medical-chatbot .
docker run -p 5000:5000 -e ACTIVELOOP_TOKEN=your-token medical-chatbot
```

### Production Considerations

1. **Use Production-Grade Server**: Replace Flask development server with Gunicorn
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 chatbot_flask:app
   ```

