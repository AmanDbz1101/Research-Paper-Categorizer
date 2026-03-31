from __future__ import annotations

import json
import os
import re
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

@dataclass
class ExtractionResult:
    title: str
    abstract: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "source": self.source,
        }


class FirstPageTitleAbstractExtractor:
    """Extracts title and abstract using first-page-first fallback up to page three."""

    def __init__(
        self,
        use_groq_fallback: bool = True,
        groq_model: str = "llama-3.3-70b-versatile",
    ) -> None:
        if not use_groq_fallback:
            raise ValueError("Heuristic-only mode is not supported. Groq extraction is mandatory.")
        self.use_groq_fallback = use_groq_fallback
        self.groq_model = groq_model

    def extract(self, pdf_path: str | Path) -> ExtractionResult:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is required. Groq extraction is mandatory.")

        title = ""
        abstract = ""

        for page_index in range(3):
            page_data = self._read_page(path, page_index)
            if not page_data:
                break

            groq_result = self._extract_with_groq(page_data["page_text"], api_key)

            candidate_title = groq_result.get("title", "").strip()
            candidate_abstract = groq_result.get("abstract", "").strip()

            if not title and self._is_valid_field(candidate_title, min_length=12):
                title = candidate_title
            if not abstract and self._is_valid_field(candidate_abstract, min_length=120):
                abstract = candidate_abstract

            if title and abstract:
                return ExtractionResult(title=title, abstract=abstract, source="groq")

        raise ValueError("Failed to extract valid title/abstract from the first 3 pages via Groq.")

    def _read_page(self, pdf_path: Path, page_index: int) -> dict[str, Any] | None:
        fitz = importlib.import_module("fitz")
        doc = fitz.open(pdf_path)
        try:
            if len(doc) == 0:
                raise ValueError("PDF has no pages")

            if page_index < 0 or page_index >= len(doc):
                return None

            page = doc[page_index]
            page_text = page.get_text("text")
            page_dict = page.get_text("dict")
            line_spans = self._flatten_line_spans(page_dict)
            return {
                "page_text": page_text,
                "line_spans": line_spans,
                "page_height": float(page.rect.height),
            }
        finally:
            doc.close()

    def _flatten_line_spans(self, page_dict: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                text_fragments = []
                max_size = 0.0

                for span in line.get("spans", []):
                    text = (span.get("text") or "").strip()
                    if not text:
                        continue
                    text_fragments.append(text)
                    size = float(span.get("size", 0.0))
                    if size > max_size:
                        max_size = size

                line_text = " ".join(text_fragments).strip()
                if not line_text:
                    continue

                bbox = line.get("bbox", [0, 0, 0, 0])
                rows.append(
                    {
                        "text": line_text,
                        "size": max_size,
                        "y": float(bbox[1]) if len(bbox) > 1 else 0.0,
                    }
                )

        rows.sort(key=lambda r: r["y"])
        return rows

    def _extract_title_heuristic(self, line_spans: list[dict[str, Any]], page_height: float) -> str:
        if not line_spans:
            return ""

        # Find lines in the top region (0-45% of page)
        top_region = [row for row in line_spans if row["y"] <= page_height * 0.45]
        if not top_region:
            top_region = line_spans[:15]

        # Filter by line length (titles are typically 8-260 chars)
        filtered = [row for row in top_region if 8 <= len(row["text"]) <= 260]
        if not filtered:
            filtered = top_region

        # Find the best (largest) font size in the region
        best_size = max((row["size"] for row in filtered), default=0.0)
        
        # Use a more lenient size cutoff: best_size - 2.0 (more forgiving)
        size_cutoff = max(best_size - 2.0, 0.0)

        # Get candidates with font size >= cutoff
        candidates = [row for row in filtered if row["size"] >= size_cutoff]
        if not candidates:
            # Fallback: take the largest font sizes available
            candidates = sorted(filtered, key=lambda r: -r["size"])[:5]
        
        if not candidates:
            return ""

        candidates.sort(key=lambda r: r["y"])

        # Collect title lines within 100 pixels (more lenient than 70)
        title_lines = []
        first_y = candidates[0]["y"]
        for row in candidates:
            if row["y"] - first_y > 100:
                break
            text = row["text"]
            if self._looks_non_title_line(text):
                continue
            title_lines.append(text)

        title = " ".join(title_lines).strip()
        title = re.sub(r"\s+", " ", title)
        return title

    def _extract_abstract_heuristic(self, page_text: str) -> str:
        clean_text = re.sub(r"\r", "\n", page_text)

        patterns = [
            re.compile(
                r"(?is)\babstract\b\s*[:\-]?\s*(.+?)(?=\n\s*(?:keywords?|index terms?|1\.?\s+introduction|introduction)\b|$)",
            ),
            re.compile(
                r"(?is)^\s*abstract\s*\n+(.+?)(?=\n\s*(?:keywords?|index terms?|1\.?\s+introduction|introduction)\b|$)",
            ),
        ]

        for pattern in patterns:
            match = pattern.search(clean_text)
            if match:
                abstract = re.sub(r"\s+", " ", match.group(1)).strip()
                return self._trim_abstract(abstract)

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", clean_text) if p.strip()]
        for paragraph in paragraphs:
            lower = paragraph.lower()
            if "abstract" in lower:
                maybe = re.sub(r"(?is)^\s*abstract\s*[:\-]?\s*", "", paragraph)
                maybe = re.sub(r"\s+", " ", maybe).strip()
                if len(maybe) > 80:
                    return self._trim_abstract(maybe)

        return ""

    def _extract_with_groq(self, page_text: str, api_key: str) -> dict[str, str]:
        groq_module = importlib.import_module("groq")
        client = groq_module.Groq(api_key=api_key)

        prompt = (
            "You are extracting metadata from the first page text of an academic research paper.\n\n"
            "What to extract:\n"
            "1) title: The paper title. It is usually near the top of the first page, often the largest and most prominent line(s), and appears before author names, affiliations, or emails.\n"
            "2) abstract: The summary paragraph under a heading like 'Abstract'. It usually appears before 'Keywords', 'Index Terms', or 'Introduction'.\n\n"
            "Rules:\n"
            "- Ignore author names, affiliations, emails, footers, page numbers, conference headers, and copyright text.\n"
            "- If title spans multiple lines, merge into one clean sentence.\n"
            "- For abstract, return only the abstract body text, not the word 'Abstract'.\n"
            "- Keep original wording; only normalize spacing.\n"
            "- Return strict JSON only with keys: title, abstract.\n"
            "- If either cannot be confidently found, return an empty string for that key.\n\n"
            f"FIRST_PAGE_TEXT:\n{page_text[:12000]}"
        )

        response = client.chat.completions.create(
            model=self.groq_model,
            temperature=0,
            max_tokens=700,
            messages=[
                {
                    "role": "system",
                    "content": "You are an information extraction engine. Output strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
            return {
                "title": str(data.get("title", "")).strip(),
                "abstract": str(data.get("abstract", "")).strip(),
            }
        except json.JSONDecodeError:
            return {"title": "", "abstract": ""}

    def _is_confident(self, title: str, abstract: str) -> bool:
        if len(title.strip()) < 12:
            return False
        if len(abstract.strip()) < 120:
            return False
        if title.strip().lower().startswith("abstract"):
            return False
        return True

    def _is_valid_field(self, field: str, min_length: int = 1) -> bool:
        """Check if a field is valid (non-empty and meets minimum length)."""
        field_clean = field.strip()
        if len(field_clean) < min_length:
            return False
        if field_clean.lower().startswith("abstract"):
            return False
        return True

    def _looks_non_title_line(self, text: str) -> bool:
        t = text.lower().strip()
        # Only filter obvious non-title markers
        bad_prefixes = (
            "abstract",
            "keywords",
            "index terms",
        )
        if t.startswith(bad_prefixes):
            return True
        # Don't filter "introduction" or "arxiv" as these could appear in titles
        if "@" in t and "." in t and len(t) < 20:
            # Likely an email, filter it
            return True
        if len(t) < 6:  # Very short lines are unlikely to be titles
            return True
        return False

    def _trim_abstract(self, abstract: str) -> str:
        if not abstract:
            return ""
        abstract = re.sub(r"\s+", " ", abstract).strip()
        if len(abstract) > 2500:
            abstract = abstract[:2500].rsplit(" ", 1)[0].strip()
        return abstract
