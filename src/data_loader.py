### RAG pipeline - Data Ingestion to Vector DB pipline
import os
import re
import trafilatura
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def process_all_txts(txt_directory):
    """Process all .txt files in a directory"""
    all_documents = []
    txt_dir = Path(txt_directory)

    # Find all .txt files recursively
    txt_files = list(txt_dir.glob("**/*.txt"))

    print(f"Found {len(txt_files)} txt files to process")

    for txt_file in txt_files:
        print(f"\nProcessing: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8", autodetect_encoding=True)
            documents = loader.load()

            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = txt_file.name
                doc.metadata['file_type'] = 'txt'

            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} document(s)")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

### Read all the pdf's inside the directory
def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

def _extract_page_text(html):
    """Extract readable text from a page's HTML.

    These are structured info pages (rosters, specs, contacts) whose key data
    is often short link text — e.g. staff names live in <a> tags. Trafilatura
    discards such anchors as "navigation", so we use BeautifulSoup get_text
    after dropping non-content tags. This keeps everything (names, titles,
    contacts) at the cost of some bounded footer noise.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    # ORNL/Drupal theme puts the big instrument mega-menu, breadcrumb and
    # skip-links in divs (not <nav>), so they survive the tag pass above and
    # otherwise pollute every page's chunks. Drop them by their stable classes.
    for sel in [".tb-megamenu", ".breadcrumb", "#skip-link",
                ".region-header", ".region-footer", ".feed-icons"]:
        for el in soup.select(sel):
            el.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Collapse runs of blank lines left behind by stripped tags.
    return re.sub(r"\n{2,}", "\n", text).strip()


### Load one or more webpages
def process_all_urls(urls):
    """Process a list of webpage URLs.

    Each page is downloaded with trafilatura and its readable text extracted
    with BeautifulSoup (see _extract_page_text for why), without needing
    per-site tag configuration.
    """
    all_documents = []

    # Accept a single URL string or a list of URLs
    if isinstance(urls, str):
        urls = [urls]

    print(f"Found {len(urls)} URLs to process")

    for url in urls:
        print(f"\nProcessing: {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                raise ValueError("failed to download page")

            text = _extract_page_text(downloaded)
            if not text:
                raise ValueError("no content extracted")

            doc = Document(
                page_content=text,
                metadata={'source_file': url, 'file_type': 'web'},
            )
            all_documents.append(doc)
            print(f"  ✓ Extracted {len(text)} chars")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents