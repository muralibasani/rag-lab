import base64
import os
import asyncio
import ssl
import nltk
import concurrent.futures
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredMarkdownLoader
)
from langchain_community.document_loaders.git import GitLoader
from langchain_core.documents import Document

from langchain_text_splitters.character import RecursiveCharacterTextSplitter


from playwright.async_api import async_playwright
from dotenv import load_dotenv
from atlassian import Jira

load_dotenv()

# --------------------------------------------
# Configuration
# --------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 400))
USER_AGENT = os.getenv("USER_AGENT", "AIAssistant/1.0 (+https://localhost)")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
resources_dir: str = "resources"

try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except Exception:
    pass  # Safe fallback if not needed

# Auto-download tokenizer data if missing
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        print(f"🔽 Downloading missing NLTK resource: {resource}")
        nltk.download(resource)


# --------------------------------------------
# HTML Cleaner
# --------------------------------------------
def clean_html_text(html_content: str) -> str:
    """Strip HTML tags, scripts, and noisy text."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text

# --------------------------------------------
# Local File Loader
# --------------------------------------------
def load_local_files(files_dirs: List[str] = None, resources_dir: str = "resources") -> List[Document]:
    """
    Load documents from local files (PDF, TXT, CSV, MD) in the given list of directories.
    Each directory is prefixed with the base resources path.
    Skips missing or unsupported files gracefully.
    """
    docs = []

    # Base resources path
    BASE_DIR = Path(__file__).resolve().parent.parent / resources_dir

    if not files_dirs or len(files_dirs) == 0:
        print(f"⚠️ No directories provided, returning empty list.")
        return docs

    for dir_path in files_dirs:
        # Prefix with base resources path
        dir_path = BASE_DIR / dir_path
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"⚠️ Directory not found or invalid, skipping: {dir_path}")
            continue

        for file_path in dir_path.iterdir():
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                elif ext in [".txt", ".text"]:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                elif ext in [".md", ".markdown"]:
                    loader = UnstructuredMarkdownLoader(str(file_path))
                elif ext == ".csv":
                    loader = CSVLoader(file_path=str(file_path))
                else:
                    print(f"⚠️ Unsupported file type ({ext}), skipping: {file_path}")
                    continue

                loaded_docs = loader.load()
                # print(f"✅ Loaded {len(loaded_docs)} document(s) from: {file_path.name}")
                docs.extend(loaded_docs)

                # Optional preview of first 500 characters
                # for i, doc in enumerate(loaded_docs, start=1):
                #     preview = doc.page_content[:500]
                #     print(f"--- Preview of document {i} from {file_path.name} ---")
                #     print(preview)
                #     print("-" * 80)

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue

    print(f"📚 Total documents loaded: {len(docs)}")
    return docs



# --------------------------------------------
# Async Web Loader (Playwright)
# --------------------------------------------
async def fetch_url_text_playwright(url: str) -> Document:
    """Fetch a fully rendered webpage with Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent=USER_AGENT)
        try:
            await page.goto(url, timeout=60000)
            await page.wait_for_load_state("networkidle")
            content = await page.content()
        finally:
            await browser.close()

    text = clean_html_text(content)
    return Document(page_content=text, metadata={"source": url})


async def load_web_links_async(urls: List[str]) -> List[Document]:
    """Fetch multiple web pages concurrently using Playwright."""
    tasks = [fetch_url_text_playwright(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    docs = []
    for url, res in zip(urls, results):
        print('Loaded URL:', url)
        if isinstance(res, Exception):
            print(f"❌ Failed to load {url}: {res}")
        else:
            docs.append(res)
            # content_length = len(res.page_content)
            # preview = res.page_content[:500]  # first 500 characters
            # print(f"✅ Loaded {url} | {content_length} chars")
            # print("Preview:")
            # print(preview)
            # print("-" * 80)  # separator
    return docs


def load_web_links(urls: List[str]) -> List[Document]:
    """Safely run Playwright async fetching in any environment (script, Jupyter, FastAPI)."""
    async def runner():
        return await load_web_links_async(urls)

    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # We're in an event loop, so we need to run this in a thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, runner())
            return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(runner())


# --------------------------------------------
# URL File Loader
# --------------------------------------------
def load_urls_from_file(url_file: str) -> List[str]:
    """Read URLs line-by-line from a file."""
    if not os.path.exists(url_file):
        return []
    with open(url_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_jira_links(urls: List[str]) -> List[Document]:
    docs = []
    
    try:
        jira = Jira(
            url="https://yourorg.atlassian.net/",
            username='yourusername',
            password=JIRA_API_TOKEN
        )

        # issue = jira.issue('FLEET-6044')

        start_at = 0
        batch_size = 50
        total_fetched = 0
        board_id = 141
        max_issues = 100

        while True:
            all_issues = jira.get_issues_for_board(
                board_id,
                jql=None,
                fields="*all",
                start=start_at,
                limit=batch_size
            )

            issues = all_issues.get("issues", [])
            if not issues:
                break

            for issue_dict in issues:
                try:
                    key = issue_dict.get("key", "")
                    print('key is ', key)
                    fields = issue_dict.get("fields", {})

                    summary = fields.get("summary", "")
                    description = fields.get("description", "")
                    issue_type = fields.get("issuetype", {}).get("name", "")
                    priority = fields.get("priority", {}).get("name", "")
                    status = fields.get("status", {}).get("name", "")
                    comments = [c.get("body", "") for c in fields.get("comment", {}).get("comments", [])]

                    content_parts = [
                        f"Issue: {key}",
                        f"FLEET Issue {key} {key} {key}",
                        f"Type: {issue_type}",
                        f"Priority: {priority}",
                        f"Status: {status}",
                        f"Summary: {summary}",
                        f"Description: {description}"
                    ] + comments

                    content = "\n\n".join([c.strip() for c in content_parts if c.strip()])
                    if not content:
                        continue

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": f"https://yourorg.atlassian.net/browse/{key}",
                            "issue_key": key,
                            "project": "Streaming Fleet",
                            "type": issue_type,
                            "priority": priority,
                            "status": status
                        },
                    )
                    docs.append(doc)
                    total_fetched += 1

                    if total_fetched >= max_issues:
                        return docs

                except Exception as e:
                    print(f"⚠️ Failed to process issue {issue_dict.get('key', '')}: {e}")

            # Prepare next batch
            start_at += batch_size
            if start_at >= all_issues.get("total", 0):
                break

        print(f"📋 Total Jira issues loaded: {len(docs)}")
        return docs
    except Exception as e:
        print(f"❌ Jira connection failed: {e}")

    print(f"📋 Total Jira issues loaded: {len(docs)}")
    return docs

# --------------------------------------------
# Unified Loader
# --------------------------------------------
def load_and_split(source_urls=[], source_files=[],
                   assistant_source_info_type=None, assistant_name=None) -> List[Document]:
    chunk_size=CHUNK_SIZE
    chunk_overlap=CHUNK_OVERLAP
    
    """Load local + web docs, clean, split into chunks."""
    # file_paths = file_paths or []
    all_docs = []

    max_files = 400

    def make_limited_file_filter(max_files: int):
        count = 0
        allowed_exts = (".md")

        def file_filter(file_path: str) -> bool:
            nonlocal count
            if not file_path.endswith(allowed_exts):
                return False
            if count >= max_files:
                return False
            count += 1
            return True

        return file_filter


    if assistant_source_info_type == "jira":
        print('Handle jira')
        all_docs = load_jira_links(source_urls)
    elif assistant_source_info_type == "github":   
        clone_url = source_urls[0]     
        clean_name = assistant_name.replace(" ", "_")
        repo_path = "/tmp/repo/" + clean_name
        repo_path_check = Path(repo_path)
        
        # Only clone if repo_path doesn't exist
        if repo_path_check.exists() and any(repo_path_check.iterdir()):
            print(f"✅ Repository already exists at {repo_path}, skipping clone.")
            loader = GitLoader(repo_path=str(repo_path), branch="main", file_filter=make_limited_file_filter(max_files))
        else:
            loader = GitLoader(
                repo_path=str(repo_path),
                clone_url=clone_url,
                branch="main",
                file_filter=make_limited_file_filter(max_files)
            )

        all_docs = loader.load()
        all_docs = all_docs[:200]
        print("📄 Loaded Git repo files...")
    else:
        if source_files and len(source_files) > 0:
            print("📄 Loading local files...")
            all_docs.extend(load_local_files(source_files))

        if source_urls and len(source_urls) > 0:
            print(f"🧾 Loading URLs from file: {source_urls}")
            # urls = load_urls_from_file(url_file)
            if source_urls and len(source_urls) > 0:
                new_docs = load_web_links(source_urls)
                all_docs.extend(new_docs)

    print(f"✅ Loaded {len(all_docs)} raw documents.")

    if len(all_docs) < 1500:
        chunk_size = 800
    else:
        chunk_size = 1200

    chunk_overlap = 150

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not all_docs:
        print("⚠️ No documents loaded, returning empty list.")
        return []
    try:
        chunks = splitter.split_documents(all_docs)
    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        return []

    print(f"✂️ Split into {len(chunks)} chunks.")
    return chunks
