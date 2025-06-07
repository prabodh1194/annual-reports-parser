from __future__ import annotations

from typing import Any
import openai
import logging

from embeddings.embed_doc import query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatClient:
    def __init__(
        self,
        vllm_base_url: str = "http://emb.openn.ai:8001/v1",
        similarity_threshold: float = 0.7,
        max_context_tokens: int = 4000,
    ) -> None:
        """
        Initialize RAG Chat Client

        Args:
            vllm_base_url: Base URL for your vLLM service
            similarity_threshold: Minimum similarity score for documents
            max_context_tokens: Maximum tokens for context (rough estimate)
        """
        # Initialize OpenAI client pointing to vLLM
        self.client = openai.OpenAI(
            base_url=vllm_base_url,
            api_key="dummy-key",  # vLLM doesn't require real API key
        )

        self.similarity_threshold = similarity_threshold
        self.max_context_tokens = max_context_tokens

        logger.info(f"Initialized RAG client with vLLM at {vllm_base_url}")

    def retrieve_documents(self, query_str: str, n_results: int = 5) -> list[dict]:
        results = query(query=query_str, n_results=n_results)

        # Filter by similarity threshold if distances are available
        filtered_docs = []

        for i in range(len(results["documents"][0])):
            doc_data = {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
                if results["metadatas"][0]
                else {},
                "id": results["ids"][0][i],
            }
            filtered_docs.append(doc_data)
            break

        logger.info(f"Retrieved {len(filtered_docs)} documents for query")
        return filtered_docs

    def format_context(self, documents: list[dict]) -> str:
        context_parts = []

        for i, doc in enumerate(documents, 1):
            doc_section = f"--- Document {i} ---\n"

            # Add metadata if available
            metadata = doc.get("metadata", {})
            if metadata:
                if "source" in metadata:
                    doc_section += f"Source: {metadata['source']}\n"
                if "title" in metadata:
                    doc_section += f"Title: {metadata['title']}\n"
                if doc.get("similarity"):
                    doc_section += f"Relevance: {doc['similarity']:.2f}\n"

            doc_section += f"Content: {doc['content']}\n\n"
            context_parts.append(doc_section)

        return "".join(context_parts)

    def truncate_context(self, context: str) -> str:
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = len(context) // 4

        if estimated_tokens > self.max_context_tokens:
            # Truncate to approximately max_context_tokens
            char_limit = self.max_context_tokens * 4
            truncated = context[:char_limit]

            # Try to cut at a document boundary
            last_doc_marker = truncated.rfind("--- Document")
            if last_doc_marker > char_limit * 0.7:  # If we can save at least 30%
                truncated = truncated[:last_doc_marker]

            logger.warning(
                f"Context truncated from ~{estimated_tokens} to ~{len(truncated) // 4} tokens"
            )
            return truncated + "\n\n[... Context truncated for length ...]"

        return context

    def create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create a RAG prompt with context and query

        Args:
            query: User query
            context: Retrieved document context

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question accurately and comprehensively.

Context:
{context}

Instructions:
- Base your answer primarily on the provided context
- If you reference information from the context, mention which document(s) you're using
- If the context doesn't contain sufficient information to answer the question, state this clearly
- Be accurate and cite your sources when possible
- Provide a helpful and complete response

Question: {query}

Answer:"""

        return prompt

    def chat_completion(
        self,
        query: str,
        model: str = "/home/ubuntu/deepseek_model",  # Adjust to your model
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> dict[str, Any]:
        # Retrieve relevant documents
        documents = self.retrieve_documents(query)

        # Format context
        context = self.format_context(documents)
        context = self.truncate_context(context)

        # Create RAG prompt
        prompt = self.create_rag_prompt(query, context)

        # Use chat completion
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": documents,
            "context_used": len(documents) > 0,
            "model": model,
            "usage": response.usage.model_dump()
            if hasattr(response, "usage")
            else None,
        }

    def chat(self, query: str, **kwargs) -> str:
        """
        Simple chat interface that returns just the answer

        Args:
            query: User query
            **kwargs: Additional arguments for chat_completion

        Returns:
            Generated answer string
        """
        result = self.chat_completion(query, **kwargs)
        return str(result["answer"])


# Example usage
if __name__ == "__main__":
    # Initialize the RAG client
    rag_client = RAGChatClient()

    # Example queries
    queries = [
        "What is the main topic discussed in the documents?",
        "Can you explain the key concepts mentioned?",
        "What are the important details I should know?",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 50)

        # Get detailed response
        result = rag_client.chat_completion(q, temperature=0.3)

        print(f"Answer: {result['answer']}")
