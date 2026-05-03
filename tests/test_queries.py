"""
Comprehensive query tests for the ConversationAgent.

Run with:
    python tests/test_queries.py
    python tests/test_queries.py --verbose    # show full responses
    python tests/test_queries.py --fast       # skip initialization (assumes model cached)
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from agents.orchestrator import Orchestrator
from agents.data_pipeline_agent import DataPipelineAgent
from agents.model_agent import ModelAgent
from agents.analytics_agent import AnalyticsAgent
from agents.conversation_agent import ConversationAgent, _extract_account_id, _extract_top_n
from agents.curveball_agent import CurveballAgent


# ─────────────────────────────────────────────────────────────────────
# Test definition
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Case:
    """Represents a single test case: a query string, a list of check functions, and a description."""

    query: str
    checks: list[Callable[[str], bool]] = field(default_factory=list)
    description: str = ""

    def run(self, response: str) -> tuple[bool, list[str]]:
        """Run all check functions against response.

        Args:
            response: The response string returned by the conversation agent.

        Returns:
            A tuple of (passed: bool, failures: list[str]) where failures contains
            a message for each check that did not pass.
        """
        failures = []
        for check in self.checks:
            try:
                if not check(response):
                    failures.append(f"  FAIL: {check.__name__ if hasattr(check,'__name__') else repr(check)}")
            except Exception as exc:
                failures.append(f"  ERROR in check: {exc}")
        return len(failures) == 0, failures


def contains(*words):
    """Return a check function that passes if all words appear in the response (case-insensitive).

    Args:
        *words: One or more substrings that must all be present in the response.

    Returns:
        A callable that accepts a response string and returns bool.
    """
    def _check(resp: str) -> bool:
        lower = resp.lower()
        return all(w.lower() in lower for w in words)
    _check.__name__ = f"contains({', '.join(words)})"
    return _check

def not_contains(*words):
    """Return a check function that passes if none of the words appear in the response.

    Args:
        *words: One or more substrings that must all be absent from the response.

    Returns:
        A callable that accepts a response string and returns bool.
    """
    def _check(resp: str) -> bool:
        lower = resp.lower()
        return not any(w.lower() in lower for w in words)
    _check.__name__ = f"not_contains({', '.join(words)})"
    return _check

def has_account_id(account_id: str):
    """Return a check function that passes if the account ID appears in the response.

    Args:
        account_id: The account identifier string to search for (case-insensitive).

    Returns:
        A callable that accepts a response string and returns bool.
    """
    def _check(resp: str) -> bool:
        return account_id.upper() in resp.upper()
    _check.__name__ = f"has_account_id({account_id})"
    return _check

def has_probability():
    """Return a check function that passes if the response contains a percentage value.

    Returns:
        A callable that accepts a response string and returns bool.
    """
    import re
    def _check(resp: str) -> bool:
        return bool(re.search(r'\d+\.?\d*\s*%', resp))
    _check.__name__ = "has_probability"
    return _check

def has_numbered_list(min_items: int = 2):
    """Return a check function that passes if response contains at least min_items numbered lines.

    Args:
        min_items: Minimum number of numbered list items required (default 2).

    Returns:
        A callable that accepts a response string and returns bool.
    """
    import re
    def _check(resp: str) -> bool:
        items = re.findall(r'^\s*\d+\.', resp, re.MULTILINE)
        return len(items) >= min_items
    _check.__name__ = f"has_numbered_list(min={min_items})"
    return _check

def no_error():
    """Return a check function that passes if response does not contain 'not found', 'error', or 'not initialized'.

    Returns:
        A callable that accepts a response string and returns bool.
    """
    def _check(resp: str) -> bool:
        lower = resp.lower()
        return "not found" not in lower and "error" not in lower and "not initialized" not in lower
    _check.__name__ = "no_error"
    return _check


# ─────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────

# Pick two real account IDs from the data (will be verified at runtime)
ACC_A = "ACC000123"
ACC_B = "ACC000456"
ACC_C = "ACC000874"   # known high-risk from earlier testing

CASES: list[Case] = [

    # ── Verbatim queries from the project brief ────────────────────────
    Case(
        query=f"Why is account {ACC_A} predicted to churn?",
        checks=[no_error(), has_account_id(ACC_A), has_probability(),
                contains("factor", "risk")],
        description="Brief example 1: why is account X at risk",
    ),
    Case(
        query=f"What are the top 3 factors driving churn risk for {ACC_A}?",
        checks=[no_error(), has_account_id(ACC_A), has_probability(),
                contains("factor")],
        description="Brief example 2: top 3 factors for account",
    ),
    Case(
        query="Which features are most important overall?",
        checks=[no_error(), contains("feature"), has_numbered_list(3),
                not_contains("not found")],
        description="Brief example 3: global feature importance",
    ),
    Case(
        query=f"What's the churn probability for account {ACC_B}?",
        checks=[no_error(), has_account_id(ACC_B), has_probability()],
        description="Brief example 4: churn probability for account",
    ),
    Case(
        query="Show me the top 5 accounts most likely to churn",
        checks=[no_error(), has_numbered_list(5), has_probability(),
                contains("churn")],
        description="Brief example 5: top 5 accounts",
    ),

    # ── Variations of explain-account ─────────────────────────────────
    Case(
        query=f"Why is {ACC_A} at risk?",
        checks=[no_error(), has_account_id(ACC_A), has_probability()],
        description="Variation: abbreviated 'why is X at risk'",
    ),
    Case(
        query=f"Explain the churn prediction for {ACC_A}",
        checks=[no_error(), has_account_id(ACC_A), has_probability()],
        description="Variation: 'explain prediction'",
    ),
    Case(
        query=f"What's driving {ACC_A} to churn?",
        checks=[no_error(), has_account_id(ACC_A), has_probability()],
        description="Variation: 'what's driving X to churn'",
    ),
    Case(
        query=f"What are the key factors for {ACC_C}?",
        checks=[no_error(), has_account_id(ACC_C), has_probability()],
        description="Variation: 'key factors for X'",
    ),
    Case(
        query=f"Break down the risk for {ACC_A}",
        checks=[no_error(), has_account_id(ACC_A), has_probability()],
        description="Variation: 'break down the risk'",
    ),
    Case(
        query=f"What are the main reasons {ACC_B} might churn?",
        checks=[no_error(), has_account_id(ACC_B), has_probability()],
        description="Variation: 'main reasons X might churn'",
    ),
    Case(
        query=f"Top 5 factors for {ACC_A}",
        checks=[no_error(), has_account_id(ACC_A), has_probability()],
        description="Variation: 'top 5 factors for X' (no why)",
    ),
    Case(
        query=f"{ACC_A}",
        checks=[no_error(), has_account_id(ACC_A), has_probability()],
        description="Variation: bare account ID",
    ),

    # ── Variations of probability queries ─────────────────────────────
    Case(
        query=f"What is the churn risk score for {ACC_B}?",
        checks=[no_error(), has_account_id(ACC_B), has_probability()],
        description="Variation: 'risk score for X'",
    ),
    Case(
        query=f"How likely is {ACC_C} to churn?",
        checks=[no_error(), has_account_id(ACC_C), has_probability()],
        description="Variation: 'how likely is X to churn'",
    ),
    Case(
        query=f"Churn probability {ACC_A}",
        checks=[no_error(), has_account_id(ACC_A), has_probability()],
        description="Variation: minimal 'churn probability X'",
    ),
    Case(
        query=f"What's the predicted churn for {ACC_B}?",
        checks=[no_error(), has_account_id(ACC_B), has_probability()],
        description="Variation: 'predicted churn for X'",
    ),

    # ── Variations of top-risk queries ────────────────────────────────
    Case(
        query="Show top 10 at-risk accounts",
        checks=[no_error(), has_numbered_list(5), has_probability()],
        description="Variation: 'show top 10 at-risk'",
    ),
    Case(
        query="Which accounts are most likely to churn?",
        checks=[no_error(), has_probability()],
        description="Variation: 'which accounts most likely to churn'",
    ),
    Case(
        query="List the top 3 accounts about to churn",
        checks=[no_error(), has_numbered_list(3), has_probability()],
        description="Variation: 'list top 3 about to churn'",
    ),
    Case(
        query="Who is at highest risk of churning?",
        checks=[no_error(), has_probability()],
        description="Variation: 'who is at highest risk'",
    ),
    Case(
        query="Give me the 7 riskiest accounts",
        checks=[no_error(), has_numbered_list(5), has_probability()],
        description="Variation: 'give me the 7 riskiest'",
    ),

    # ── Feature importance variations ──────────────────────────────────
    Case(
        query="What features predict churn?",
        checks=[no_error(), has_numbered_list(3)],
        description="Variation: 'what features predict churn'",
    ),
    Case(
        query="What are the top predictors of churn?",
        checks=[no_error(), has_numbered_list(3)],
        description="Variation: 'top predictors of churn'",
    ),
    Case(
        query="Which features drive churn overall?",
        checks=[no_error(), has_numbered_list(3)],
        description="Variation: 'which features drive churn'",
    ),
    Case(
        query="What drives customer churn in our data?",
        checks=[no_error(), has_numbered_list(3)],
        description="Variation: 'what drives customer churn'",
    ),

    # ── Model performance variations ──────────────────────────────────
    Case(
        query="How is the model performing?",
        checks=[no_error(), contains("auc", "recall")],
        description="Variation: 'how is the model performing'",
    ),
    Case(
        query="What's the model accuracy?",
        checks=[no_error(), contains("accuracy")],
        description="Variation: 'model accuracy'",
    ),
    Case(
        query="How accurate is the churn prediction?",
        checks=[no_error(), contains("auc")],
        description="Variation: 'how accurate is prediction'",
    ),

    # ── Summary variations ─────────────────────────────────────────────
    Case(
        query="Give me a portfolio summary",
        checks=[no_error(), contains("total", "account", "risk")],
        description="Variation: 'portfolio summary'",
    ),
    Case(
        query="How many accounts are high risk?",
        checks=[no_error(), contains("high", "risk")],
        description="Variation: 'how many accounts are high risk'",
    ),
    Case(
        query="Overview of all accounts",
        checks=[no_error(), contains("total", "account")],
        description="Variation: 'overview of all accounts'",
    ),

    # ── Edge cases ─────────────────────────────────────────────────────
    Case(
        query="ACC999999",   # non-existent account
        checks=[contains("not found")],
        description="Edge: non-existent account ID",
    ),
    Case(
        query="",
        checks=[contains("please enter")],
        description="Edge: empty query",
    ),
    Case(
        query="/help",
        checks=[contains("acc000", "top", "feature", "help")],
        description="Slash command: /help",
    ),
    Case(
        query="/curveball list",
        checks=[contains("column_rename", "new_source", "curveball")],
        description="Slash command: /curveball list",
    ),
]


# ─────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────

def build_orch() -> Orchestrator:
    """Construct and return a fully wired Orchestrator with all agents registered; used by run_tests.

    Returns:
        An Orchestrator instance with DataPipelineAgent, ModelAgent, AnalyticsAgent,
        ConversationAgent, and CurveballAgent all registered.
    """
    config = load_config()
    orch = Orchestrator(config)
    orch.register(DataPipelineAgent(orch))
    orch.register(ModelAgent(orch))
    orch.register(AnalyticsAgent(orch))
    orch.register(ConversationAgent(orch))
    orch.register(CurveballAgent(orch))
    return orch


def run_tests(verbose: bool = False) -> None:
    """Initialize the orchestrator, run all test Cases, print pass/fail results, and exit 0 if all pass.

    Args:
        verbose: If True, print the query and truncated response for every test case.

    Returns:
        None (exits the process with code 0 on full pass, 1 on any failure).
    """
    print("=" * 70)
    print("ChurnGuard — Conversation Query Test Suite")
    print("=" * 70)

    print("\nInitializing pipeline…")
    t0 = time.time()
    orch = build_orch()
    orch.initialize()
    print(f"Ready in {time.time()-t0:.1f}s\n")

    passed = 0
    failed = 0
    errors = []

    for i, case in enumerate(CASES, 1):
        try:
            resp = orch.send("conversation", "query", {"text": case.query})
            response = resp.get("response", resp.get("message", ""))
            ok, failures = case.run(response)
        except Exception as exc:
            ok = False
            failures = [f"  EXCEPTION: {exc}"]
            response = ""

        status = "✓" if ok else "✗"
        desc = case.description or case.query[:60]
        print(f"  {status} [{i:>2}] {desc}")

        if not ok:
            failed += 1
            errors.append((case, failures, response))
            for f in failures:
                print(f"        {f}")
            if verbose:
                print(f"        Query:    {case.query!r}")
                print(f"        Response: {response[:300]!r}")
        else:
            passed += 1
            if verbose:
                print(f"        Query:    {case.query!r}")
                print(f"        Response: {response[:200]!r}")

    print()
    print("─" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(CASES)} tests")

    if errors:
        print("\nFailed cases:")
        for case, failures, response in errors:
            print(f"\n  Query: {case.query!r}")
            print(f"  Response: {response[:200]!r}")
            for f in failures:
                print(f"  {f}")

    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_tests(verbose=args.verbose)
