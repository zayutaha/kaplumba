"""Research memory — tracks what's been searched, found, and covered."""

from __future__ import annotations
from dataclasses import dataclass, field


DIMENSION_TEMPLATES = {
    "person": [
        "biography", "timeline", "career", "major_actions",
        "relationships", "beliefs_or_ideology", "controversies",
        "criticism", "achievements", "legacy", "historical_interpretation",
    ],
    "place": [
        "geography", "history", "demographics", "economy",
        "culture", "government", "infrastructure", "tourism",
        "politics", "current_events",
    ],
    "event": [
        "background", "timeline", "causes", "actors",
        "key_moments", "outcomes", "consequences", "criticism",
        "competing_interpretations", "legacy",
    ],
    "organization": [
        "history", "structure", "leadership", "mission",
        "activities", "impact", "controversies", "membership",
        "funding", "criticism",
    ],
    "technology": [
        "overview", "history", "how_it_works", "applications",
        "comparisons", "limitations", "impact", "future",
        "ecosystem", "criticism",
    ],
    "scientific_concept": [
        "definition", "history", "mechanism", "evidence",
        "applications", "related_concepts", "debates",
        "open_questions", "impact",
    ],
    "general_topic": [
        "overview", "history", "key_ideas", "major_figures",
        "controversies", "current_state", "impact", "criticism",
        "future_outlook",
    ],
}


@dataclass
class ResearchMemory:
    topic: str = ""
    topic_type: str = "general_topic"
    dimensions: dict = field(default_factory=dict)
    searched_queries: list = field(default_factory=list)
    entities_seen: list = field(default_factory=list)
    concepts_seen: list = field(default_factory=list)
    candidate_docs: list = field(default_factory=list)
    accepted_urls: set = field(default_factory=set)
    novelty_history: list = field(default_factory=list)
    iteration: int = 0

    @property
    def avg_coverage(self) -> float:
        if not self.dimensions:
            return 0.0
        return sum(self.dimensions.values()) / len(self.dimensions)

    def get_weak_dimensions(self, threshold: float = 0.5) -> list[str]:
        return [d for d, v in self.dimensions.items() if v < threshold]

    def get_strong_dimensions(self, threshold: float = 0.7) -> list[str]:
        return [d for d, v in self.dimensions.items() if v >= threshold]

    def have_seen_url(self, url: str) -> bool:
        return url in self.accepted_urls

    def have_seen_text(self, text: str) -> bool:
        lower = text.lower()
        for doc in self.candidate_docs:
            if doc.get("title", "").lower() in lower or doc.get("snippet", "").lower() in lower:
                return True
        return False
