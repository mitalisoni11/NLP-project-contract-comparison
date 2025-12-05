"""
External Agent - Performs real semantic search on Pinecone using embedded WIPO documents.
"""

import os
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
        self._openai_client: Optional[OpenAI] = None

        # Configurable settings
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
        else:
            # Fallback: create from config (backward compatibility)
            from ..services.llm_factory import create_llm_service
            self._llm = create_llm_service()
        
        # Override configurable settings if provided
        if "index_name" in config:
            self.INDEX_NAME = config["index_name"]
        if "embed_model" in config:
            self.EMBED_MODEL = config["embed_model"]

        # Pinecone init from environment variable or config
        pinecone_api_key = config.get("pinecone_api_key") or os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("pinecone_api_key is required for ExternalAgent (set PINECONE_API_KEY env var or pass in config)")
        
        self._pinecone_client = Pinecone(api_key=pinecone_api_key)
        self._pinecone_index = self._pinecone_client.Index(self.INDEX_NAME)
        
        # Initialize OpenAI client from environment variable or config
        openai_api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("openai_api_key is required for ExternalAgent (set OPENAI_API_KEY env var or pass in config)")
        
        self._openai_client = OpenAI(api_key=openai_api_key)

        logger.info("ExternalAgent initialized successfully.")

    # ---------------------------- SEARCH ----------------------------
    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for the semantic search query."""
        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialized. Call initialize() first.")
        
        resp = self._openai_client.embeddings.create(
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
    def _get_llm_service(self, request: Dict[str, Any]) -> LLMService:
        """Get LLM service from request override or use default."""
        if "llm_service" in request:
            return request["llm_service"]
        return self._llm

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic search requests."""
        command = request.get("command")

        if command != "query":
            return {"success": False, "error": f"Unknown command: {command}"}

        query = request.get("query", "")
        if not query:
            return {"success": False, "error": "Query is required"}
        
        # Get model override and LLM service from request if provided
        model_override = request.get("model")
        llm_service_to_use = self._get_llm_service(request)
        
        # Handle selected_documents parameter (for consistency with InternalAgent)
        selected_documents = request.get("selected_documents", [])
        if selected_documents:
            logger.info(f"      📄 Selected documents passed to ExternalAgent: {selected_documents}")
            logger.info(f"      ℹ️  Note: ExternalAgent queries Pinecone, not document files")

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
        llm_response = await llm_service_to_use.generate(
            system=self._system_prompt,
            prompt=f"""
User query: {query}

Retrieved context:
{combined_context}
""",
            max_tokens=300,
            model=model_override
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
