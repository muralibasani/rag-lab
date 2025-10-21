# AI Assistants

Leverage AI, RAG, LangChain, Ollama, and FAISS embeddings to set up this project in your organization for efficient internal documentation search.
All processing happens locally.

You can index internal documents in .txt, .pdf, .md, .csv formats, as well as content from URLs (internal or external), GitHub repositories, or Jira issues.

## 🚀 Quick Installation

### Option 1: Automated Installation (Recommended)

```bash
# Clone the repository

# Run the automated installation script
source install.sh
```

### Option 2: Manual Installation
  
#### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
# or
venv\Scripts\activate      # Windows
```

#### 2. Install Dependencies

**Modern approach (recommended):**
```bash
pip install --upgrade pip
pip install -e .
```

**Traditional approach:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Development setup:**
```bash
pip install --upgrade pip
pip install -e .[dev]
```

#### 3. Install Playwright Browsers

```bash
playwright install chromium
```

#### 4. Install Ollama LLM Locally

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

#### 5. Download the Local Model

```bash
ollama pull llama3
```

#### 6. Start the Ollama Server

```bash
ollama serve
```

## 🏃‍♂️ Running the Application

### Start the App

python3 app.py

## 🧠 Vectordb: Embedding Storage Options

The assistant supports two vector database backends for storing document embeddings:

### 1. FAISS (In-Memory)
- **Default option** for fast, temporary storage.
- All embeddings are stored in RAM and lost when the backend restarts.
- Use for quick experiments or when persistence is not required.
- Set in `.env`:
  ```env
  VECTOR_EMBEDDINGS_STORAGE=inmemory
  ```

### 2. Chroma (Persistent)
- Stores embeddings on disk for long-term use.
- Embeddings are saved in the directory specified by `VECTOR_PERSIST_DIR` (default: `./vector_store`).
- Recommended for production or when you want to retain embeddings between runs.
- Set in `.env`:
  ```env
  VECTOR_EMBEDDINGS_STORAGE=chroma
  VECTOR_PERSIST_DIR=vector_store
  ```
- The Chroma DB is automatically created if it does not exist, or loaded if present.

#### configuring assistants.json
- assistant_name Name of assistant
- source_info_type - jira/github/text
- model - llama3/granite
- source_files - directory inside resources dir with all local files
- source_urls -  all urls array

#### Troubleshooting Chroma
- If you see errors about missing or corrupted Chroma DB, delete the `chroma_store` directory and restart the backend to rebuild it.
- Always use the same embedding model for both creation and querying.

#### Switching Vector Stores
- Change the `VECTOR_EMBEDDINGS_STORAGE` value in your `.env` file to switch between FAISS and Chroma.
- Restart the backend after changing this setting.

#### Best Practices
- For persistent, scalable QA, use Chroma.
- For quick tests or development, use FAISS.

## Notes on Usage and Licensing

This project runs entirely **on your local machine**.  
It uses the **Ollama Llama 3** model, downloaded and executed locally — no data or queries are sent to external cloud services (such as OpenAI or Anthropic). All computation, vector storage, and inference happen on your own system.

⚠️ **Important:**  
Do **not** include or scrape other web sources without verifying their license or obtaining explicit permission.  
This ensures your local assistant remains compliant with open-source and fair-use principles.

#### Useful links
https://reference.langchain.com/python/langchain/
