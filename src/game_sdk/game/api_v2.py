import requests
from typing import List
import time

from typing import List, Dict

class GAMEClientV2:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://sdk.game.virtuals.io/v2"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }

    def create_agent(self, name: str, description: str, goal: str) -> str:
        """
        API call to create an agent instance (worker or agent with task generator)
        """
        payload = {
            "data": {
                "name": name,
                "goal": goal,
                "description": description
            }
        }

        response = requests.post(
            f"{self.base_url}/agents",
            headers=self.headers,
            json=payload
        )

        response_json = response.json()
        if response.status_code != 200:
            raise ValueError(f"Failed to post data: {response_json}")

        return response_json["data"]["id"]

    def create_workers(self, workers: List) -> str:
        """
        API call to create workers and worker description for the task generator (agent)
        """
        payload = {
            "data": {
                "locations": [
                    {"id": w.id, "name": w.id, "description": w.worker_description}
                    for w in workers
                ]
            }
        }

        response = requests.post(
            f"{self.base_url}/maps",
            headers=self.headers,
            json=payload
        )

        response_json = response.json()
        if response.status_code != 200:
            raise ValueError(f"Failed to post data: {response_json}")

        return response_json["data"]["id"]

    def set_worker_task(self, agent_id: str, task: str) -> Dict:
        """
        API call to set worker task (for standalone worker)
        """
        payload = {
            "data": {
                "task": task
            }
        }

        response = requests.post(
            f"{self.base_url}/agents/{agent_id}/tasks",
            headers=self.headers,
            json=payload
        )

        response_json = response.json()
        if response.status_code != 200:
            raise ValueError(f"Failed to post data: {response_json}")

        return response_json["data"]

    def get_worker_action(self, agent_id: str, submission_id: str, data: dict) -> Dict:
        """
        API call to get worker actions (for standalone worker)
        """
        response = requests.post(
            f"{self.base_url}/agents/{agent_id}/tasks/{submission_id}/next",
            headers=self.headers,
            json={
                "data": data
            }
        )

        response_json = response.json()
        if response.status_code != 200:
            raise ValueError(f"Failed to post data: {response_json}")

        return response_json["data"]

    def get_agent_action(self, agent_id: str, data: dict) -> Dict:
        """
        API call to get agent actions/next step (for agent)
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/agents/{agent_id}/actions",
                    headers=self.headers,
                    json={
                        "data": data
                    }
                )

                response_json = response.json()
                if response.status_code != 200:
                    if "Concurrent request detected" in str(response_json):
                        if attempt < max_retries - 1:  # Don't sleep on last attempt
                            time.sleep(retry_delay)
                            continue
                    raise ValueError(f"Failed to post data: {response_json}")

                return response_json["data"]
                
            except Exception as e:
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    time.sleep(retry_delay)
                    continue
                raise e

        return {"action": {"type": "do_nothing"}}  # Fallback response if all retries fail