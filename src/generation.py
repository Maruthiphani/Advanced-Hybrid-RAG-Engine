"""Grounded answer generation with inline [1][2]-style citations.

`MockGenerationProvider` is an extractive QA system, not a placeholder --
extractive QA (pull the most relevant sentence per source rather than
generate new prose) is a legitimate, well-established approach in its own
right, and it has a structural property generative answers don't: it is
*incapable* of hallucinating a claim the source text doesn't contain,
because every word in its answer is copied, not generated. `OpenAIProvider`
/ `AnthropicProvider` produce real synthesized prose under a system prompt
that requires citing every claim and explicitly permits "I don't know."
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import List, Tuple

from src.models import Citation, RetrievalCandidate
from src.text_utils import overlap_ratio

SYSTEM_PROMPT = """\
You are a documentation assistant. Answer the user's question using ONLY the numbered \
source passages provided below.

Rules:
- Cite the source for every factual claim using its marker, e.g. [1], [2]. A claim with \
no citation is not acceptable.
- If the passages don't contain enough information to answer, say so explicitly rather \
than guessing or using outside knowledge.
- Do not cite a source for something it doesn't actually say.
- Be concise and direct.
"""


class GenerationProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, question: str, candidates: List[RetrievalCandidate]) -> Tuple[str, List[Citation]]: ...


def _build_citations(candidates: List[RetrievalCandidate]) -> List[Citation]:
    return [
        Citation(marker=f"[{i}]", chunk_id=c.chunk.id, source_file=c.chunk.source_file, section_heading=c.chunk.section_heading)
        for i, c in enumerate(candidates, start=1)
    ]


class MockGenerationProvider(GenerationProvider):
    name = "mock-extractive"

    def generate(self, question: str, candidates: List[RetrievalCandidate]) -> Tuple[str, List[Citation]]:
        citations = _build_citations(candidates)
        if not candidates:
            return "I don't have enough information in the indexed documentation to answer this.", []

        sentences_used = []
        for citation, candidate in zip(citations, candidates):
            sentences = re.split(r"(?<=[.!?])\s+", candidate.chunk.text.strip())
            sentences = [s for s in sentences if len(s.split()) >= 4]
            if not sentences:
                continue
            best = max(sentences, key=lambda s: overlap_ratio(question, s))
            sentences_used.append(f"{best.strip()} {citation.marker}")

        if not sentences_used:
            return "I don't have enough information in the indexed documentation to answer this.", []

        answer = " ".join(sentences_used)
        return answer, citations


def _format_context(candidates: List[RetrievalCandidate], citations: List[Citation]) -> str:
    blocks = []
    for citation, candidate in zip(citations, candidates):
        heading = f" ({candidate.chunk.section_heading})" if candidate.chunk.section_heading else ""
        blocks.append(f"{citation.marker} Source: {candidate.chunk.source_file}{heading}\n{candidate.chunk.text}")
    return "\n\n".join(blocks)


class OpenAIGenerationProvider(GenerationProvider):
    name = "openai"

    def __init__(self, api_key: str = None, model: str = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model or os.environ.get("GENERATION_MODEL", "gpt-4o-mini")

    def generate(self, question: str, candidates: List[RetrievalCandidate]) -> Tuple[str, List[Citation]]:
        citations = _build_citations(candidates)
        context = _format_context(candidates, citations)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content, citations


class AnthropicGenerationProvider(GenerationProvider):
    name = "anthropic"

    def __init__(self, api_key: str = None, model: str = None):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    def generate(self, question: str, candidates: List[RetrievalCandidate]) -> Tuple[str, List[Citation]]:
        citations = _build_citations(candidates)
        context = _format_context(candidates, citations)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"}],
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        return text, citations


def get_generation_provider(name: str = None) -> GenerationProvider:
    name = name or os.environ.get("GENERATION_PROVIDER")
    if name == "openai":
        return OpenAIGenerationProvider()
    if name == "anthropic":
        return AnthropicGenerationProvider()
    if name == "mock":
        return MockGenerationProvider()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIGenerationProvider()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicGenerationProvider()
    return MockGenerationProvider()
