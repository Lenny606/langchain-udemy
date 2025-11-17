from langchain_classic.callbacks.base import BaseCallbackHandler
from typing import Dict, Any, List


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any

                     ) -> Any:
        """Runs when the LLM starts."""
        print(f"LLM is starting... prompt: {prompts[0]}")

    def on_llm_end(self, response, **kwargs: Any) -> Any:
        """Runs when the LLM ends."""
        print(f"LLM is Ending... prompt: {response.generations[0][0].text}")
