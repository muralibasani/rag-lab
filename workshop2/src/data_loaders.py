import os
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
     TextLoader, DirectoryLoader
)
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from playwright.async_api import async_playwright
from dotenv import load_dotenv

from src.models import EventType

load_dotenv()

# --------------------------------------------
# Configuration
# --------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 400))
USER_AGENT = os.getenv("USER_AGENT", "AIAssistant/1.0 (+https://localhost)")
KAFKA_DOCS_URL = os.getenv("KAFKA_DOCS_URL")
resources_dir: str = "resources"


def get_log_file_pattern(event_type: EventType) -> str:
    log_file_pattern = ""

    if event_type == EventType.KAFKA:
        log_file_pattern = "server"
    elif event_type == EventType.SCHEMA_REGISTRY:
        log_file_pattern = "schema"

    return log_file_pattern

def load_local_files(event_type : EventType) -> List[Document]:
    BASE_DIR = Path(__file__).resolve().parent.parent / resources_dir
    dir_path = BASE_DIR / 'logs'
    log_file_pattern = get_log_file_pattern(event_type)
    loader = DirectoryLoader(
        dir_path,
        glob=f"**/{log_file_pattern}.log*",   # matches .log, .log.2025-10-27, etc.
        loader_cls=TextLoader,
        show_progress=True
    )

    raw_docs = loader.load()

    print(f"✅ Loaded {len(raw_docs)} raw documents from: {dir_path}")
    return raw_docs


def clean_html_text(html_content: str) -> str:
    """Strip HTML tags, scripts, and noisy text."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text

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


# --------------------------------------------
# Unified Loader
# --------------------------------------------
def load_and_split(event_type:EventType) -> List[Document]:
    chunk_size=CHUNK_SIZE
    chunk_overlap=CHUNK_OVERLAP
    
    """Load local + web docs, clean, split into chunks."""
    # file_paths = file_paths or []
    all_docs = []

    if event_type == EventType.KAFKA or event_type == EventType.SCHEMA_REGISTRY:
        all_docs.extend(load_local_files(event_type))
        chunk_size = 500
        chunk_overlap = 75
    elif event_type == EventType.KAFKA_DOCS :
        print(f"🧾 Loading URL")
        # urls = load_urls_from_file(url_file)
        new_docs = load_web_links([KAFKA_DOCS_URL])
        all_docs.extend(new_docs)
        chunk_size = 1000
        chunk_overlap = 150

    # print(f"✅ Loaded {len(all_docs)} raw documents.")

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
