"""
External Agent - Performs real semantic search on Pinecone using embedded WIPO documents.
"""

from uuid import UUID
from typing import Dict, Any, List, Optional
import logging

from ..interfaces.agent import AgentInterface
from ..services.llm_service import LLMService
from pinecone import Pinecone
from openai import OpenAI

logger = logging.getLogger(__name__)


class ExternalAgent(AgentInterface):
    """Agent that performs real semantic search over Pinecone index (WIPO documents)."""

    def __init__(self):
        self._uuid: UUID = None
        self._llm: Optional[LLMService] = None
        self._pinecone_client: Optional[Pinecone] = None
        self._pinecone_index = None

        # You can configure this
        self.INDEX_NAME = "wipo-index"
        self.EMBED_MODEL = "text-embedding-3-small"

        # System prompt instructing the LLM how to behave
        self._system_prompt = """
You are an external WIPO document agent.

You MUST:
- Use ONLY the information retrieved from Pinecone semantic search
- Cite the filenames and chunk numbers exactly
- Produce concise factual summaries
- Avoid hallucinating sections not present in retrieved text

Response format:
1. Short answer (3–6 sentences)
2. "Sources:" list filenames + chunk_ids
"""

    # Required interface properties
    @property
    def name(self) -> str:
        return "External Agent"

    @property
    def description(self) -> str:
        return "Performs semantic search on WIPO documents using Pinecone."

    @property
    def agent_id_str(self) -> str:
        return "external_agent"

    @property
    def uuid(self) -> UUID:
        return self._uuid

    @uuid.setter
    def uuid(self, value: UUID):
        self._uuid = value

    # ---------------------------- INIT ----------------------------
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize LLM and Pinecone from config."""
        # Accept injected LLMService
        if "llm_service" in config:
            self._llm = config["llm_service"]

        # Pinecone init
        if "pinecone_api_key" in config:
            self._pinecone_client = Pinecone(api_key=config["pinecone_api_key"])
            self._pinecone_index = self._pinecone_client.Index(self.INDEX_NAME)

        else:
            raise ValueError("pinecone_api_key is required for ExternalAgent")

        logger.info("ExternalAgent initialized successfully.")

    # ---------------------------- SEARCH ----------------------------
    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for the semantic search query."""
        client = OpenAI()

        resp = client.embeddings.create(
            model=self.EMBED_MODEL,
            input=query
        )

        return resp.data[0].embedding

    def _semantic_search(self, query_embed: List[float], top_k=5):
        """Run a Pinecone similarity search."""
        result = self._pinecone_index.query(
            vector=query_embed,
            top_k=top_k,
            include_metadata=True
        )
        return result.matches or []

    # ---------------------------- MAIN HANDLER ----------------------------
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic search requests."""
        command = request.get("command")

        if command != "query":
            return {"success": False, "error": f"Unknown command: {command}"}

        query = request.get("query", "")
        if not query:
            return {"success": False, "error": "Query is required"}

        logger.info("🔵 ExternalAgent: Embedding query...")
        query_embed = self._embed_query(query)

        logger.info("🔍 ExternalAgent: Running semantic search...")
        matches = self._semantic_search(query_embed=query_embed, top_k=5)

        if not matches:
            return {
                "success": True,
                "data": {
                    "query": query,
                    "response": "No relevant WIPO documents found.",
                    "source": []
                }
            }

        # Aggregate retrieved text
        combined_context = "\n\n".join(
            f"[{m.metadata['file_name']} - chunk {m.metadata['chunk_id']}]\n{m.metadata['text']}"
            for m in matches
        )

        # LLM final answer
        logger.info("🧠 ExternalAgent: Calling LLM for summary...")
        llm_response = await self._llm.generate(
            system=self._system_prompt,
            prompt=f"""
User query: {query}

Retrieved context:
{combined_context}
""",
            max_tokens=300,
        )

        return {
            "success": True,
            "data": {
                "query": query,
                "response": llm_response,
                "source": [
                    {
                        "file_name": m.metadata["file_name"],
                        "chunk_id": m.metadata["chunk_id"],
                        "score": m.score,
                    }
                    for m in matches
                ]
            }
        }

    # ---------------------------- STATUS & TOOLS ----------------------------
    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "query",
                "description": "Query Pinecone-indexed WIPO documents",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "User query for semantic search"
                    }
                }
            }
        ]

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "active",
            "pinecone_connected": self._pinecone_index is not None,
            "llm_connected": self._llm is not None
        }

    async def shutdown(self) -> None:
        pass
