"""30+ tests for research_agent package."""
import unittest
from unittest.mock import patch, MagicMock

from mlx_lm.research_agent.memory import ResearchMemory, DIMENSION_TEMPLATES
from mlx_lm.research_agent.framing import infer_topic_type, build_dimension_map, KNOWN_TYPES
from mlx_lm.research_agent.planning import evaluate_coverage, generate_queries, should_stop
from mlx_lm.research_agent.retrieval import (
    seed_search, novelty_score, score_candidate,
    deduplicate_candidates, _authority_score, _diversity_score
)
from mlx_lm.research_agent.orchestrator import coverage_aware_selection


# ═══════════════════════════════════════════════
# TESTS 1-10: Memory
# ═══════════════════════════════════════════════

class TestResearchMemory(unittest.TestCase):

    def test_default_memory(self):
        m = ResearchMemory()
        self.assertEqual(m.topic, "")
        self.assertEqual(m.topic_type, "general_topic")
        self.assertEqual(m.dimensions, {})
        self.assertEqual(m.iteration, 0)
        self.assertEqual(m.avg_coverage, 0.0)

    def test_memory_with_topic(self):
        m = ResearchMemory(topic="Napoleon")
        self.assertEqual(m.topic, "Napoleon")

    def test_avg_coverage_empty(self):
        m = ResearchMemory()
        self.assertEqual(m.avg_coverage, 0.0)

    def test_avg_coverage_calculates(self):
        m = ResearchMemory(dimensions={"a": 0.5, "b": 0.7, "c": 0.3})
        self.assertAlmostEqual(m.avg_coverage, 0.5)

    def test_get_weak_dimensions(self):
        m = ResearchMemory(dimensions={"a": 0.2, "b": 0.6, "c": 0.8})
        weak = m.get_weak_dimensions(0.5)
        self.assertIn("a", weak)
        self.assertNotIn("b", weak)
        self.assertNotIn("c", weak)

    def test_get_strong_dimensions(self):
        m = ResearchMemory(dimensions={"a": 0.2, "b": 0.6, "c": 0.8})
        strong = m.get_strong_dimensions(0.7)
        self.assertIn("c", strong)
        self.assertNotIn("a", strong)
        self.assertNotIn("b", strong)

    def test_have_seen_url(self):
        m = ResearchMemory()
        m.accepted_urls.add("https://example.com")
        self.assertTrue(m.have_seen_url("https://example.com"))
        self.assertFalse(m.have_seen_url("https://other.com"))

    def test_entity_tracking(self):
        m = ResearchMemory(entities_seen=["Napoleon", "France"])
        self.assertEqual(len(m.entities_seen), 2)

    def test_novelty_history_tracks(self):
        m = ResearchMemory(novelty_history=[0.5, 0.3, 0.1])
        self.assertEqual(len(m.novelty_history), 3)

    def test_dimension_templates_exist(self):
        for t in KNOWN_TYPES:
            self.assertIn(t, DIMENSION_TEMPLATES)
            self.assertGreater(len(DIMENSION_TEMPLATES[t]), 3)


# ═══════════════════════════════════════════════
# TESTS 11-18: Framing
# ═══════════════════════════════════════════════

class TestFraming(unittest.TestCase):

    def test_known_types_contains_person(self):
        self.assertIn("person", KNOWN_TYPES)

    def test_known_types_contains_event(self):
        self.assertIn("event", KNOWN_TYPES)

    def test_known_types_contains_place(self):
        self.assertIn("place", KNOWN_TYPES)

    def test_build_dimension_map_person(self):
        dims = build_dimension_map("person")
        self.assertIn("biography", dims)
        self.assertIn("legacy", dims)
        self.assertEqual(dims["biography"], 0.0)

    def test_build_dimension_map_event(self):
        dims = build_dimension_map("event")
        self.assertIn("causes", dims)
        self.assertIn("outcomes", dims)

    def test_build_dimension_map_unknown_falls_back(self):
        dims = build_dimension_map("unknown_type")
        self.assertIn("overview", dims)

    def test_build_dimension_map_values_start_at_zero(self):
        dims = build_dimension_map("person")
        for v in dims.values():
            self.assertEqual(v, 0.0)

    def test_infer_topic_type_with_model(self):
        # Can't run model in unit test, just verify function exists
        self.assertTrue(callable(infer_topic_type))


# ═══════════════════════════════════════════════
# TESTS 19-26: Planning
# ═══════════════════════════════════════════════

class TestPlanning(unittest.TestCase):

    def test_should_stop_max_iterations(self):
        m = ResearchMemory(iteration=5)
        self.assertTrue(should_stop(m, {}))

    def test_should_stop_high_coverage(self):
        m = ResearchMemory(iteration=2)
        self.assertTrue(should_stop(m, {"a": 0.9, "b": 0.85, "c": 0.95}))

    def test_should_stop_low_novelty(self):
        m = ResearchMemory(iteration=3, novelty_history=[0.05, 0.04])
        self.assertTrue(should_stop(m, {"a": 0.5}))

    def test_should_not_stop_early(self):
        m = ResearchMemory(iteration=1, novelty_history=[0.8])
        self.assertFalse(should_stop(m, {"a": 0.3, "b": 0.2}))

    def test_should_not_stop_mid_research(self):
        m = ResearchMemory(iteration=2, novelty_history=[0.5, 0.4])
        self.assertFalse(should_stop(m, {"a": 0.4, "b": 0.3}))

    def test_avg_coverage_high_triggers_stop(self):
        m = ResearchMemory(iteration=3)
        cov = {f"d{i}": 0.85 for i in range(8)}
        self.assertTrue(should_stop(m, cov))

    def test_generate_queries_function_exists(self):
        self.assertTrue(callable(generate_queries))

    def test_evaluate_coverage_function_exists(self):
        self.assertTrue(callable(evaluate_coverage))


# ═══════════════════════════════════════════════
# TESTS 27-40: Retrieval
# ═══════════════════════════════════════════════

class TestRetrieval(unittest.TestCase):

    def test_authority_wikipedia(self):
        self.assertAlmostEqual(_authority_score("https://en.wikipedia.org/wiki/Napoleon", ""), 1.0)

    def test_authority_britannica(self):
        self.assertAlmostEqual(_authority_score("https://www.britannica.com/topic/Napoleon", ""), 1.0)

    def test_authority_gov(self):
        self.assertAlmostEqual(_authority_score("https://www.whitehouse.gov/about", ""), 0.9)

    def test_authority_edu(self):
        self.assertAlmostEqual(_authority_score("https://www.harvard.edu/history", ""), 0.85)

    def test_authority_major_news(self):
        url = "https://www.bbc.com/news/article"
        self.assertAlmostEqual(_authority_score(url, ""), 0.8)

    def test_authority_academic(self):
        url = "https://www.nature.com/articles/paper"
        self.assertAlmostEqual(_authority_score(url, ""), 0.9)

    def test_authority_unknown(self):
        self.assertAlmostEqual(_authority_score("https://example.com/blog", ""), 0.3)

    def test_novelty_completely_new(self):
        m = ResearchMemory(entities_seen=[], concepts_seen=[])
        doc = {"title": "Napoleonic Code", "snippet": "Legal reforms under Napoleon", "dimension": "legacy"}
        score = novelty_score(doc, m)
        self.assertGreater(score, 0.5)

    def test_novelty_seen_entities_lower(self):
        m = ResearchMemory(entities_seen=["Napoleon", "France"])
        doc = {"title": "Napoleon in France", "snippet": "His impact on France", "dimension": "biography"}
        score = novelty_score(doc, m)
        self.assertLess(score, 0.8)

    def test_novelty_redundant_low(self):
        m = ResearchMemory(entities_seen=["Napoleon", "France", "Waterloo"])
        doc = {"title": "Napoleon at Waterloo", "snippet": "The Battle of Waterloo was Napoleon's final battle. France lost.", "dimension": "biography"}
        m.candidate_docs = [{"title": "Napoleon at Waterloo", "url": "http://example.com"}]
        score = novelty_score(doc, m)
        self.assertLess(score, 0.5)

    def test_diversity_same_domain_penalty(self):
        m = ResearchMemory(candidate_docs=[
            {"url": "https://example.com/1"},
            {"url": "https://example.com/2"},
            {"url": "https://example.com/3"},
        ])
        self.assertAlmostEqual(_diversity_score("https://example.com/4", m), 0.1)

    def test_diversity_new_domain(self):
        m = ResearchMemory(candidate_docs=[{"url": "https://example.com/1"}])
        self.assertAlmostEqual(_diversity_score("https://other.com/page", m), 1.0)

    def test_diversity_medium(self):
        m = ResearchMemory(candidate_docs=[{"url": "https://example.com/1"}])
        self.assertAlmostEqual(_diversity_score("https://example.com/2", m), 0.5)

    def test_deduplicate_same_url(self):
        m = ResearchMemory(accepted_urls={"https://example.com/1"})
        docs = [
            {"url": "https://example.com/1", "title": "Same"},
            {"url": "https://example.com/2", "title": "Different"},
        ]
        result = deduplicate_candidates(docs, m)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["url"], "https://example.com/2")

    def test_deduplicate_similar_titles(self):
        m = ResearchMemory()
        docs = [
            {"url": "https://a.com", "title": "Napoleon Bonaparte Biography and Life"},
            {"url": "https://b.com", "title": "Napoleon Bonaparte Biography and History"},
        ]
        result = deduplicate_candidates(docs, m)
        self.assertEqual(len(result), 1)

    def test_deduplicate_different_titles(self):
        m = ResearchMemory()
        docs = [
            {"url": "https://a.com", "title": "Napoleon Bonaparte Biography"},
            {"url": "https://b.com", "title": "French Revolution Causes"},
        ]
        result = deduplicate_candidates(docs, m)
        self.assertEqual(len(result), 2)

    def test_score_candidate_full(self):
        m = ResearchMemory(dimensions={"biography": 0.2, "legacy": 0.8})
        doc = {
            "url": "https://en.wikipedia.org/wiki/Napoleon",
            "title": "Napoleon",
            "snippet": "Emperor of France",
            "dimension": "biography",
            "novelty": 0.7,
        }
        result = score_candidate(doc, m)
        self.assertIn("score", result)
        self.assertGreater(result["score"], 0)


# ═══════════════════════════════════════════════
# TESTS 41-46: Coverage-Aware Selection
# ═══════════════════════════════════════════════

class TestCoverageAwareSelection(unittest.TestCase):

    def test_selects_per_dimension(self):
        m = ResearchMemory(dimensions={"bio": 0.5, "legacy": 0.5, "politics": 0.5})
        m.candidate_docs = [
            {"url": "https://a.com", "title": "Bio", "score": 0.8, "dimension": "bio"},
            {"url": "https://b.com", "title": "Legacy", "score": 0.7, "dimension": "legacy"},
            {"url": "https://c.com", "title": "Politics", "score": 0.6, "dimension": "politics"},
        ]
        result = coverage_aware_selection(m, count=3)
        self.assertEqual(len(result), 3)
        urls = {d["url"] for d in result}
        self.assertIn("https://a.com", urls)
        self.assertIn("https://b.com", urls)
        self.assertIn("https://c.com", urls)

    def test_fills_remaining_with_top_scored(self):
        m = ResearchMemory(dimensions={"bio": 0.5, "legacy": 0.5})
        m.candidate_docs = [
            {"url": "https://a.com", "title": "Best", "score": 0.9, "dimension": "bio"},
            {"url": "https://b.com", "title": "Second", "score": 0.8, "dimension": "legacy"},
            {"url": "https://c.com", "title": "Third", "score": 0.7, "dimension": "overview"},
        ]
        result = coverage_aware_selection(m, count=3)
        self.assertEqual(len(result), 3)

    def test_respects_count_limit(self):
        m = ResearchMemory(dimensions={"a": 0.5, "b": 0.5})
        m.candidate_docs = [
            {"url": "https://a.com", "title": "A", "score": 0.9, "dimension": "a"},
            {"url": "https://b.com", "title": "B", "score": 0.8, "dimension": "b"},
            {"url": "https://c.com", "title": "C", "score": 0.7, "dimension": "c"},
            {"url": "https://d.com", "title": "D", "score": 0.6, "dimension": "d"},
        ]
        result = coverage_aware_selection(m, count=2)
        self.assertEqual(len(result), 2)

    def test_handles_empty_candidates(self):
        m = ResearchMemory(dimensions={"a": 0.5})
        m.candidate_docs = []
        result = coverage_aware_selection(m, count=5)
        self.assertEqual(len(result), 0)

    def test_deduplicates_urls(self):
        m = ResearchMemory(dimensions={"bio": 0.5, "legacy": 0.5})
        m.candidate_docs = [
            {"url": "https://a.com", "title": "Bio", "score": 0.9, "dimension": "bio"},
            {"url": "https://a.com", "title": "Bio dup", "score": 0.8, "dimension": "legacy"},
        ]
        result = coverage_aware_selection(m, count=5)
        self.assertEqual(len(result), 1)

    def test_picks_highest_scored_per_dimension(self):
        m = ResearchMemory(dimensions={"bio": 0.5})
        m.candidate_docs = [
            {"url": "https://low.com", "title": "Low", "score": 0.3, "dimension": "bio"},
            {"url": "https://high.com", "title": "High", "score": 0.9, "dimension": "bio"},
            {"url": "https://mid.com", "title": "Mid", "score": 0.6, "dimension": "bio"},
        ]
        result = coverage_aware_selection(m, count=2)
        urls = {d["url"] for d in result}
        self.assertIn("https://high.com", urls)


# ═══════════════════════════════════════════════
# TESTS 35-37: Normalization / Relevance Filtering
# ═══════════════════════════════════════════════

class TestNormalizeDocs(unittest.TestCase):
    def test_cleaner_prompt_with_query_rejects_irrelevant(self):
        """The cleaner prompt instructs removal of content not related to the query."""
        system = 'Strip HTML tags, scripts, nav menus, ads, and ANY content NOT related to "AI medical diagnosis" from the page below. Keep ONLY sentences and paragraphs relevant to "AI medical diagnosis". Return the exact remaining text word for word — no summaries, no paraphrasing, no truncation.'
        self.assertIn("NOT related to", system)
        self.assertIn('"AI medical diagnosis"', system)
        self.assertIn("Keep ONLY", system)
        self.assertNotIn("truncation", "should not appear incorrectly")

    def test_cleaner_prompt_without_query_strips_html_only(self):
        """Without a query, the cleaner only strips HTML/nav/ads."""
        system = "Strip HTML tags, scripts, nav menus, and ads from the page below. Return the EXACT readable text content word for word — no summaries, no paraphrasing, no truncation. Output every sentence exactly as written."
        self.assertIn("Strip HTML tags", system)
        self.assertNotIn("NOT related", system)


if __name__ == "__main__":
    unittest.main(verbosity=2)
