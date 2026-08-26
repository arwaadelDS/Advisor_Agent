import sys
import os

# Add root directory to path so imports resolve cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.workflow import app
from langchain_core.messages import HumanMessage

BENCHMARK_TESTS = [
    # --- SQL / Client Portfolio Routing ---
    {
        "id": 1,
        "category": "SQL Routing",
        "input": "Show me client 101's portfolio breakdown and cash balance.",
        "expected_agent": "sql_agent",
    },
    {
        "id": 2,
        "category": "SQL Routing",
        "input": "Which clients hold shares in Saudi Aramco?",
        "expected_agent": "sql_agent",
    },
    {
        "id": 3,
        "category": "SQL Routing",
        "input": "What is the total value of client 102's asset allocation?",
        "expected_agent": "sql_agent",
    },

    # --- Live Search / External Market Routing ---
    {
        "id": 4,
        "category": "Search Routing",
        "input": "What is the latest Tadawul All Share Index (TASI) closing price today?",
        "expected_agent": "search_agent",
    },
    {
        "id": 5,
        "category": "Search Routing",
        "input": "What are the latest breaking news headlines about oil prices and OPEC+ quotas?",
        "expected_agent": "search_agent",
    },
    {
        "id": 6,
        "category": "Search Routing",
        "input": "Fetch the current USD/SAR exchange rate and central bank interest rate.",
        "expected_agent": "search_agent",
    },

    # --- Internal RAG / Research PDF Routing ---
    {
        "id": 7,
        "category": "RAG Routing",
        "input": "What is our analyst rating and target price for ACWA Power from our internal research reports?",
        "expected_agent": "rag_agent",
    },
    {
        "id": 8,
        "category": "RAG Routing",
        "input": "Summarize the Q2 earnings analysis and risks for Al Rajhi Bank from our equity research notes.",
        "expected_agent": "rag_agent",
    },
    {
        "id": 9,
        "category": "RAG Routing",
        "input": "Find the macroeconomic sector outlook on Saudi banking published in our research library.",
        "expected_agent": "rag_agent",
    },

    # --- Multi-Agent Handoffs & Edge Cases ---
    {
        "id": 10,
        "category": "Multi-Agent Handoff",
        "input": "Retrieve client 101's stock holdings and check if we have any research notes published on those companies.",
        "expected_agent": "sql_agent",  # Supervisor should route to SQL first, then RAG
    },
    {
        "id": 11,
        "category": "Supervisor Direct / Guardrail",
        "input": "Hello! What capabilities do you have to assist financial advisors?",
        "expected_agent": "end",
    },
    {
        "id": 12,
        "category": "Supervisor Guardrail",
        "input": "Can you write a poem about artificial intelligence?",
        "expected_agent": "end",
    },
]


def run_benchmarks():
    print("\n" + "=" * 70)
    print("🚀 STARTING MULTI-AGENT BENCHMARK EVALUATION (12 QUERIES)")
    print("=" * 70 + "\n")

    passed = 0
    total = len(BENCHMARK_TESTS)

    for test in BENCHMARK_TESTS:
        print(f"[{test['id']}/{total}] Category: {test['category']}")
        print(f"Query: \"{test['input']}\"")
        print(f"Expected Target: {test['expected_agent']}")

        initial_state = {
            "messages": [HumanMessage(content=test["input"])],
            "client_id": "101" if "101" in test["input"] else ("102" if "102" in test["input"] else None),
            "sql_result": None,
            "rag_context": None,
            "search_result": None,
            "next_agent": None,
        }

        try:
            result = app.invoke(initial_state)
            last_agent = result.get("next_agent", "end")
            print(f"Resulting Route: {last_agent}")
            
            # Print agent response if available
            if result.get("messages") and len(result["messages"]) > 1:
                print(f"Response: {result['messages'][-1].content[:120]}...")

            passed += 1
            print("Status: ✅ Executed\n" + "-" * 50)
        except Exception as e:
            print(f"Status: ❌ Error during execution: {e}\n" + "-" * 50)

    print(f"\nBenchmark Run Complete: {passed}/{total} queries executed successfully.\n")


if __name__ == "__main__":
    run_benchmarks()
