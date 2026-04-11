"""
Wiki context utility for Claude Code integration.
Helps developers find relevant wiki pages when working on code.

Usage:
    from backend.wiki_context import get_wiki_context
    context = get_wiki_context("crawlers/feature_engineer.py")
    print(context.related_pages)  # List of wiki pages to read
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class WikiContext:
    """Encapsulates wiki context for a code file."""

    def __init__(self, code_file: str, map_data: Dict):
        self.code_file = code_file
        self.map_data = map_data

    @property
    def related_pages(self) -> List[str]:
        """Get list of related wiki pages for this code file."""
        mapping = self.map_data.get("mappings", {})
        for file_pattern, info in mapping.items():
            if file_pattern in self.code_file or self.code_file.endswith(file_pattern):
                return info.get("wiki_pages", [])
        return []

    @property
    def purpose(self) -> Optional[str]:
        """Get the purpose of this code file."""
        mapping = self.map_data.get("mappings", {})
        for file_pattern, info in mapping.items():
            if file_pattern in self.code_file or self.code_file.endswith(file_pattern):
                return info.get("purpose")
        return None

    def format_for_llm(self, index_data: Dict) -> str:
        """Format context for Claude to read."""
        output = []
        output.append(f"📌 Code File: {self.code_file}")
        output.append(f"📝 Purpose: {self.purpose}\n")

        output.append("📚 Related Wiki Pages:")
        for page_key in self.related_pages:
            page = index_data.get("pages", {}).get(page_key)
            if page:
                output.append(f"  • {page['title']} ({page['path']})")
                output.append(f"    {page['summary']}")

        return "\n".join(output)


def get_wiki_context(code_file: str, project_root: Optional[Path] = None) -> Optional[WikiContext]:
    """
    Get wiki context for a code file.

    Args:
        code_file: Path to code file (e.g., "crawlers/feature_engineer.py")
        project_root: Root directory (defaults to cwd)

    Returns:
        WikiContext if mapping found, else None
    """
    if project_root is None:
        project_root = Path.cwd()

    map_path = project_root / ".claude" / "code-wiki-map.json"
    if not map_path.exists():
        return None

    try:
        map_data = json.loads(map_path.read_text())
        return WikiContext(code_file, map_data)
    except Exception:
        return None


def get_wiki_index(project_root: Optional[Path] = None) -> Optional[Dict]:
    """Load the wiki index."""
    if project_root is None:
        project_root = Path.cwd()

    index_path = project_root / ".claude" / "wiki-index.json"
    if not index_path.exists():
        return None

    try:
        return json.loads(index_path.read_text())
    except Exception:
        return None


if __name__ == "__main__":
    # Example: Print wiki context for feature_engineer.py
    import sys

    code_file = sys.argv[1] if len(sys.argv) > 1 else "crawlers/feature_engineer.py"

    context = get_wiki_context(code_file)
    index = get_wiki_index()

    if context and index:
        print(context.format_for_llm(index))
    else:
        print(f"No context found for {code_file}")
