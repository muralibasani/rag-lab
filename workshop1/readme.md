Leverage AI, RAG, LangChain, Ollama, and FAISS embeddings to set up this project in your organization for efficient internal documentation search.
All processing happens locally.

You can index internal documents in .txt, .pdf, .md, .csv formats, as well as content from URLs (internal or external), GitHub repositories, or Jira issues.

## 🚀 Quick Installation

### Install backend
uv sync

Resources - 
Place a pdf file in resources dir

### Install Ollama LLM Locally

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### Download the Local Model

```bash
ollama pull llama3
```

### Start the Ollama Server

```bash
ollama serve
```

## 🏃‍♂️ Running the Application

### Start the App cli

set start_cli = True

python3 app.py

+-----------+  
| __start__ |  
+-----------+  
      *        
      *        
      *        
  +-------+    
  | model |    
  +-------+    
      *        
      *        
      *        
 +---------+   
 | __end__ |   
 +---------+   
✅ Loaded 1 document(s) from: resources/F2510149287.pdf
✂️ Split into 1 chunks.
🧠 You: what is the invoice amount

🤖 Assistant:
what is the invoice amount
Answer is  content='The invoice amount is € 16.15 (€ 13.76 + € 2.39).' additional_kwargs={} response_metadata={'model': 'llama3', 'created_at': '2025-11-08T11:18:00.174957Z', 'done': True, 'done_reason': 'stop', 'total_duration': 2130150417, 'load_duration': 796969667, 'prompt_eval_count': 425, 'prompt_eval_duration': 936165500, 'eval_count': 23, 'eval_duration': 395541500, 'model_name': 'llama3', 'model_provider': 'ollama'} id='lc_run--268abce1-ed20-4131-b1f9-da6f915dc3c4-0' usage_metadata={'input_tokens': 425, 'output_tokens': 23, 'total_tokens': 448}

🤖 Assistant:
The invoice amount is € 16.15 (€ 13.76 + € 2.39).


### With backend and front end
uvicorn app:app --reload --host 0.0.0.0 --port 8000

In fe dir :
npm install (install deps)

#### Run
npm run dev

#### Best Practices
- For persistent, scalable QA, use Chroma.
- For quick tests or development, use FAISS.

## Notes on Usage and Licensing

This project runs entirely **on your local machine**.  
It uses the **Ollama Llama 3** model, downloaded and executed locally — no data or queries are sent to external cloud services (such as OpenAI or Anthropic). All computation, vector storage, and inference happen on your own system.

⚠️ **Important:**  
Do **not** include or scrape other web sources without verifying their license or obtaining explicit permission.  
This ensures your local assistant remains compliant with open-source and fair-use principles.

## Create a front end project
Install node and npm (https://nodejs.org/en)

node -v
npm create vite@latest fe-react
Select React
Select TypeScript
cd fe-react
npm install
npm run dev

#### Useful links
https://reference.langchain.com/python/langchain/
