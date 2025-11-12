import json
import re
import time
import os
from dotenv import load_dotenv
from collections import deque
from pathlib import Path
from typing import List
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup

load_dotenv()
BOILERPLATE_LINES = {
    "You need to enable JavaScript to run this app.",
    "Log In",
    "Help Center",
    "How can we help you?",
    "Venmo",
    "Topics",
    "Resources",
    "Why Venmo",
    "Trust & safety",
    "Money Talks",
    "Our fees",
    "Developers",
    "Company",
    "About us",
    "Jobs",
    "Accessibility",
    "News & Press",
    "Blog",
    "Legal",
    "Terms",
    "Privacy",
    "Cookies",
    "Contact us",
}

BOILERPLATE_PREFIXES = (
    "Send & Receive",
    "Pay with Venmo",
    "Venmo Debit Card",
    "Venmo Credit Card",
    "Venmo for Business",
    "Accept Venmo",
    "Help Center",
)

BOILERPLATE_SUFFIXES = (
    "home_page",
    "section_page",
)

FOOTER_PATTERNS = (
    re.compile(r"^Venmo is a service of PayPal", re.IGNORECASE),
    re.compile(r"©\s?\d{4}\s?PayPal", re.IGNORECASE),
)

def scrub_lines(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            if cleaned and cleaned[-1] == "":
                continue
            cleaned.append("")
            continue
        normalized = " ".join(stripped.split())
        if normalized in BOILERPLATE_LINES:
            continue
        if any(normalized.startswith(prefix) for prefix in BOILERPLATE_PREFIXES):
            continue
        if any(normalized.endswith(suffix) for suffix in BOILERPLATE_SUFFIXES):
            continue
        if any(pattern.search(normalized) for pattern in FOOTER_PATTERNS):
            continue
        cleaned.append(stripped)
    # Trim leading/trailing blanks introduced by cleaning
    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return cleaned


def clean_content(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = scrub_lines(lines)
    # Collapse consecutive duplicate lines that may survive after scrub
    deduped: List[str] = []
    for line in cleaned_lines:
        if deduped and deduped[-1] == line:
            continue
        deduped.append(line)
    return "\n".join(deduped)


def scrape_site(start_url: str, max_pages: int = 50, delay: float = 0.5):
    """Breadth-first crawl constrained to the start domain."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; RAGBot/0.1)"})

    to_visit = deque([start_url])
    visited, documents = set(), []

    start_netloc = urlparse(start_url).netloc

    while to_visit and len(visited) < max_pages:
        url = to_visit.popleft()
        url = urldefrag(url)[0]           # drop in-page anchors

        if url in visited:
            continue
        if urlparse(url).netloc != start_netloc:
            continue

        try:
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"skip {url}: {exc}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        main = soup.find("main") or soup.body
        text = main.get_text(separator="\n", strip=True) if main else ""
        content = clean_content(text)
        if not content:
            visited.add(url)
            continue

        documents.append({"url": url, "content": content})
        visited.add(url)

        for link in soup.find_all("a", href=True):
            candidate = urljoin(url, link["href"])
            candidate = urldefrag(candidate)[0]
            if candidate not in visited:
                to_visit.append(candidate)

        time.sleep(delay)

    return documents


def save_documents(documents, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(documents, indent=2))


def load_documents(output_path: Path):
    return json.loads(output_path.read_text())


if __name__ == "__main__":
    start_url = os.getenv("SCRAPE_START_URL")
    output_path = Path(os.getenv("SCRAPE_OUTPUT_PATH"))

    if output_path.exists():
        docs = load_documents(output_path)
        print(f"Loaded {len(docs)} cached documents from {output_path}")
    else:
        docs = scrape_site(start_url, max_pages=80)
        save_documents(docs, output_path)
        print(f"Scraped and cached {len(docs)} documents to {output_path}")