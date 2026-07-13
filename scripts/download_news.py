from __future__ import annotations

import argparse
import hashlib
import html
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag_core import clean_whitespace, write_jsonl  # noqa: E402


DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/mundo/rss.xml",
    "https://rss.dw.com/xml/rss-sp-all",
    "https://news.google.com/rss/search?q=inteligencia%20artificial%20OR%20machine%20learning&hl=es-419&gl=US&ceid=US:es-419",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download real news items from RSS/Atom feeds for a RAG corpus."
    )
    parser.add_argument(
        "--feed",
        action="append",
        dest="feeds",
        help="RSS/Atom feed URL. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "data" / "news.jsonl",
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--max-items-per-feed",
        type=int,
        default=20,
        help="Maximum news items to keep from each feed.",
    )
    return parser.parse_args()


def fetch_xml(url: str) -> ET.Element:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "basic-rag-course-example/1.0",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        data = response.read()
    return ET.fromstring(data)


def strip_tags(value: str) -> str:
    text = html.unescape(value or "")
    output = []
    inside_tag = False
    for char in text:
        if char == "<":
            inside_tag = True
            output.append(" ")
            continue
        if char == ">":
            inside_tag = False
            continue
        if not inside_tag:
            output.append(char)
    return repair_mojibake(clean_whitespace("".join(output)))


def repair_mojibake(value: str) -> str:
    if "Ã" not in value and "Â" not in value:
        return value
    try:
        repaired = value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value
    return repaired


def child_text(element: ET.Element, names: list[str]) -> str:
    for child in list(element):
        local_name = child.tag.split("}")[-1].lower()
        if local_name in names:
            return repair_mojibake(clean_whitespace(child.text or ""))
    return ""


def child_attr(element: ET.Element, child_name: str, attr_name: str) -> str:
    for child in list(element):
        local_name = child.tag.split("}")[-1].lower()
        if local_name == child_name:
            return repair_mojibake(clean_whitespace(child.attrib.get(attr_name, "")))
    return ""


def feed_title(root: ET.Element, feed_url: str) -> str:
    channel = root.find("channel")
    if channel is not None:
        title = child_text(channel, ["title"])
        if title:
            return title
    title = child_text(root, ["title"])
    return title or feed_url


def rss_items(root: ET.Element) -> list[ET.Element]:
    channel = root.find("channel")
    if channel is not None:
        return channel.findall("item")
    return [
        child
        for child in list(root)
        if child.tag.split("}")[-1].lower() in {"entry", "item"}
    ]


def item_to_record(item: ET.Element, source: str, feed_url: str) -> dict[str, Any]:
    title = strip_tags(child_text(item, ["title"]))
    summary = strip_tags(child_text(item, ["description", "summary", "content", "encoded"]))
    link = child_text(item, ["link"])
    if not link:
        link = child_attr(item, "link", "href")
    published = child_text(item, ["pubdate", "published", "updated"])

    raw_id = "|".join([source, title, link])
    doc_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
    return {
        "id": doc_id,
        "source": source,
        "feed_url": feed_url,
        "title": title,
        "summary": summary,
        "published": published,
        "link": link,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def download_feed(feed_url: str, max_items: int) -> list[dict[str, Any]]:
    root = fetch_xml(feed_url)
    source = feed_title(root, feed_url)
    records = []
    for item in rss_items(root)[:max_items]:
        record = item_to_record(item, source=source, feed_url=feed_url)
        if record["title"] and (record["summary"] or record["link"]):
            records.append(record)
    return records


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for record in records:
        key = record["link"] or record["id"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def main() -> None:
    args = parse_args()
    feeds = args.feeds or DEFAULT_FEEDS
    records = []

    for feed_url in feeds:
        try:
            feed_records = download_feed(feed_url, max_items=args.max_items_per_feed)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not download {feed_url}: {exc}", file=sys.stderr)
            continue
        print(f"Downloaded {len(feed_records)} items from {feed_url}")
        records.extend(feed_records)

    records = dedupe(records)
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} unique news documents to {args.output}")


if __name__ == "__main__":
    main()
