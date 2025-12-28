#!/usr/bin/env python3
"""
Bookmark to Reading List Parser

Transforms Chrome-exported bookmarks into a JSON dataset for the knowledge graph.
Handles arXiv papers, PDFs, and web pages with robust error handling.
"""

import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Configuration
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
REQUEST_TIMEOUT = 30
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
MAX_SUMMARY_WORDS = 1000
MIN_SUMMARY_WORDS = 15


class BookmarkParser(HTMLParser):
    """Parse Chrome bookmark HTML export."""

    def __init__(self):
        super().__init__()
        self.bookmarks: List[Tuple[str, str]] = []
        self.current_link: Optional[str] = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            attrs_dict = dict(attrs)
            if "href" in attrs_dict:
                self.current_link = attrs_dict["href"]

    def handle_data(self, data):
        if self.current_link:
            title = data.strip()
            if title:
                self.bookmarks.append((title, self.current_link))
                self.current_link = None


def parse_bookmarks_file(file_path: Path) -> List[Tuple[str, str]]:
    """Parse Chrome bookmarks HTML file and extract (title, url) pairs."""
    print(f"Parsing bookmarks from: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    parser = BookmarkParser()
    parser.feed(content)

    print(f"  - Found {len(parser.bookmarks)} bookmarks")
    return parser.bookmarks


def classify_url(url: str) -> str:
    """Classify URL as 'arxiv', 'pdf', or 'web'."""
    url_lower = url.lower()

    if "arxiv.org" in url_lower:
        return "arxiv"
    if url_lower.endswith(".pdf") or "/pdf/" in url_lower:
        return "pdf"
    return "web"


def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arXiv ID from URL."""
    patterns = [
        r"arxiv\.org/abs/(\d+\.\d+)",
        r"arxiv\.org/pdf/(\d+\.\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def fetch_arxiv_metadata(arxiv_id: str) -> Optional[Dict[str, str]]:
    """Fetch arXiv paper metadata and abstract from API."""
    try:
        url = f"{ARXIV_API_BASE}?id_list={arxiv_id}"
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "xml")
        entry = soup.find("entry")

        if not entry:
            return None

        title = entry.find("title")
        summary = entry.find("summary")

        return {
            "title": title.get_text(strip=True) if title else None,
            "summary": summary.get_text(strip=True) if summary else None,
        }

    except Exception as e:
        print(f"    ⚠ arXiv API error: {e}")
        return None


def download_and_extract_pdf(url: str) -> Optional[str]:
    """Download PDF and extract text content."""
    try:
        print("    - Downloading PDF...")
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        # Write to temporary file and extract text
        reader = PdfReader(response.content)
        text_parts = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        full_text = " ".join(text_parts)
        return full_text.strip()

    except Exception as e:
        print(f"    ⚠ PDF extraction error: {e}")
        return None


def extract_web_content(url: str) -> Optional[str]:
    """Extract main content from web page."""
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Try to find main content
        main_content = None
        for tag in ["main", "article", "div[role='main']"]:
            main_content = soup.find(tag)
            if main_content:
                break

        if not main_content:
            main_content = soup.find("body")

        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
            # Clean up whitespace
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        return None

    except Exception as e:
        print(f"    ⚠ Web extraction error: {e}")
        return None


def truncate_to_words(text: str, max_words: int) -> str:
    """Truncate text to maximum number of words."""
    if not text:
        return ""

    words = text.split()
    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words])


def generate_stable_id(title: str, content: str) -> str:
    """Generate stable ID from title and content hash."""
    # Create a slug from title
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")
    slug = slug[:50]  # Limit length

    # Add content hash for uniqueness
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]

    return f"{slug}-{content_hash}"


def process_bookmark(title: str, url: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Process a single bookmark and return (result_dict, error_message).
    Returns (None, error) on failure, or (dict, None) on success.

    Failures occur when:
    - Content extraction fails
    - Summary has fewer than MIN_SUMMARY_WORDS words
    """
    print(f"\n  Processing: {title[:60]}...")
    print(f"    URL: {url}")

    resource_type = classify_url(url)
    print(f"    Type: {resource_type}")

    final_title = title
    summary = ""
    error = None

    try:
        if resource_type == "arxiv":
            arxiv_id = extract_arxiv_id(url)
            if arxiv_id:
                print(f"    arXiv ID: {arxiv_id}")
                metadata = fetch_arxiv_metadata(arxiv_id)

                if metadata:
                    if metadata["title"]:
                        final_title = metadata["title"]
                    if metadata["summary"]:
                        summary = metadata["summary"]
                else:
                    error = "Failed to fetch arXiv metadata"
            else:
                error = "Could not extract arXiv ID"

        elif resource_type == "pdf":
            pdf_text = download_and_extract_pdf(url)
            if pdf_text:
                summary = truncate_to_words(pdf_text, MAX_SUMMARY_WORDS)
            else:
                error = "Failed to extract PDF content"

        else:  # web
            web_text = extract_web_content(url)
            if web_text:
                summary = truncate_to_words(web_text, MAX_SUMMARY_WORDS)
            else:
                error = "Failed to extract web content"

        # Check for extraction errors
        if error:
            print(f"    ✗ FAILED: {error}")
            return None, error

        # Validate summary length
        word_count = len(summary.split())
        if word_count < MIN_SUMMARY_WORDS:
            error = f"Insufficient content (only {word_count} words, minimum {MIN_SUMMARY_WORDS} required)"
            print(f"    ✗ FAILED: {error}")
            return None, error

        # Generate stable ID
        content_for_id = f"{final_title}{summary}"
        stable_id = generate_stable_id(final_title, content_for_id)

        result = {
            "id": stable_id,
            "title": final_title,
            "summary": summary,
            "link": url,
        }

        print(
            f"    ✓ Successfully processed (summary: {word_count} words, {len(summary)} chars)"
        )
        return result, None

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"    ✗ FAILED: {error_msg}")
        return None, error_msg


def main():
    """Main execution flow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse Chrome bookmarks into reading list JSON"
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to Chrome bookmarks HTML export"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/reading_list.json"),
        help="Output JSON file path (default: data/reading_list.json)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Bookmark to Reading List Parser")
    print("=" * 70)

    # Parse bookmarks
    bookmarks = parse_bookmarks_file(args.input_file)

    if not bookmarks:
        print("No bookmarks found!")
        return

    # Process each bookmark
    results = []
    errors = []

    for i, (title, url) in enumerate(bookmarks, 1):
        print(f"\n[{i}/{len(bookmarks)}]")

        result, error = process_bookmark(title, url)

        if result:
            results.append(result)
        if error:
            errors.append({"title": title, "url": url, "error": error})

        # Rate limiting
        if i < len(bookmarks):
            time.sleep(args.delay)

    # Summary
    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)
    print(f"Total bookmarks: {len(bookmarks)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {len(errors)}")

    # Save results
    print(f"\nSaving to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save error log if there were errors
    if errors:
        error_log_path = args.output.parent / "parsing_errors.json"
        print(f"Saving error log to: {error_log_path}")
        with open(error_log_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
