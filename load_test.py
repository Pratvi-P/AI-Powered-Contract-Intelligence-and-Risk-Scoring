"""
Day 10: Load Testing with Locust

Tests the Contract Intelligence API under simulated concurrent users.

Install:
    pip install locust

Run (make sure uvicorn is running first on port 8000):
    locust -f load_test.py --host=http://localhost:8000

Then open: http://localhost:8089
Set number of users (try 10) and spawn rate (2 per second) -> Start

What to look for:
    - Response times stay under 2000ms
    - Failure rate stays at 0%
    - RPS (requests per second) the API can handle
"""

from locust import HttpUser, between, task


class ContractAPIUser(HttpUser):
    # Each simulated user waits 1-3 seconds between requests
    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        """Most frequent task - just checks API is alive."""
        self.client.get("/health")

    @task(2)
    def get_categories(self):
        """Get the list of 41 clause categories."""
        self.client.get("/categories")

    @task(1)
    def search_contracts(self):
        """Search for similar contracts by query."""
        self.client.post(
            "/search",
            json={
                "query": "governing law California jurisdiction",
                "top_k": 3,
            },
        )
