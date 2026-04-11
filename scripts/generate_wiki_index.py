#!/usr/bin/env python3
"""
Generate wiki-index.json for Claude Code context injection.
Runs during setup to create a searchable index of all wiki pages.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def extract_title(content: str) -> str:
    """Extract title from markdown heading."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1).strip() if match else "Untitled"


def extract_summary(content: str) -> str:
    """Extract first meaningful paragraph."""
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('>'):
            # Remove markdown formatting for summary
            summary = re.sub(r'[*_`\[\]()]', '', line)
            return summary[:100] + ("..." if len(summary) > 100 else "")
    return ""


def extract_keywords(content: str) -> List[str]:
    """Extract keywords from bold text and code blocks."""
    keywords = set()

    # Extract bold text: **word**
    for match in re.finditer(r'\*\*(.+?)\*\*', content):
        keywords.add(match.group(1).lower())

    # Extract code: `word`
    for match in re.finditer(r'`([^`]+)`', content):
        keywords.add(match.group(1).lower())

    return sorted(list(keywords))[:10]  # Top 10


def generate_index(project_root: Path) -> Dict[str, Any]:
    """Generate wiki index from all markdown files."""
    wiki_dir = project_root / "wiki"

    if not wiki_dir.exists():
        print(f"Wiki directory not found: {wiki_dir}")
        return {}

    index = {
        "version": "1.0",
        "generated_at": str(Path(project_root / ".claude").resolve()),
        "pages": {},
        "categories": {}
    }

    # Scan all markdown files
    for md_file in sorted(wiki_dir.rglob("*.md")):
        if "gpuadvicewiki" in str(md_file):
            continue  # Skip Obsidian vault folder

        try:
            content = md_file.read_text(encoding='utf-8')
            rel_path = md_file.relative_to(wiki_dir)
            category = str(rel_path.parent)

            page_key = str(rel_path).replace(".md", "").replace("/", "::")

            index["pages"][page_key] = {
                "path": f"wiki/{rel_path}",
                "title": extract_title(content),
                "summary": extract_summary(content),
                "category": category,
                "keywords": extract_keywords(content),
                "lines": len(content.split('\n'))
            }

            # Track categories
            if category != ".":
                if category not in index["categories"]:
                    index["categories"][category] = []
                index["categories"][category].append(page_key)

        except Exception as e:
            print(f"Error processing {md_file}: {e}")

    return index


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent

    index = generate_index(project_root)

    # Write index
    output_path = project_root / ".claude" / "wiki-index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))

    print(f"✓ Wiki index generated: {output_path}")
    print(f"  - Total pages: {len(index['pages'])}")
    print(f"  - Categories: {list(index['categories'].keys())}")
