import asyncio
import json
import os
import re
import time

from astabench.types.sqa import SQAResponse
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import Generate, Solver, TaskState, solver
from openai import OpenAI
from pydantic import ValidationError


async def query_openai_deep_research(
    model: str,
    question: str,
    timeout: int = 3600,
):
    """Query OpenAI Deep Research API using the Responses API.
    
    Args:
        model: Deep research model name (e.g., "o3-deep-research" or "o4-mini-deep-research")
        question: The research question to answer
        timeout: Timeout in seconds for the API call (default: 3600 for 1 hour)
    
    Returns:
        SQAResponse object with the parsed response
    """
    client = OpenAI(timeout=timeout, api_key=os.getenv("OPENAI_API_KEY"))

    prompt = (
        "Please write a report responding to the following user query. Return your result as valid JSON with a single key `sections` which is a JSON list of sections, each having keys `title`, `text`, and `citations`. Each claim in the text should be supported by an inline citation. Each entry in `citations` should have a JSON list of `snippets` extracted from the reference document and an `id`, each of which appears exactly in the text. Each `id` should be an inline citation. Any additional information about the citation should go under `metadata`.\n\n"
        + question
    )
    
    # Create deep research request with web search tool
    # Note: background defaults to False (synchronous execution)
    # The SDK may not support the background parameter yet, so we omit it
    response = client.responses.create(
        model="o3-deep-research-2025-06-26",
        input=prompt,
        tools=[
            {"type": "web_search_preview"},
        ],
    )
    
    # Poll for completion if the response is still in progress
    # This ensures we wait for the synchronous request to complete
    start_time = time.time()
    while response.status not in ("completed", "failed", "cancelled"):
        # Check timeout
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Deep Research task exceeded timeout of {timeout} seconds")
        
        if response.status == "in_progress":
            # Wait before polling again
            await asyncio.sleep(10)
            # Retrieve updated response status
            response = client.responses.retrieve(response.id)
        else:
            # Unknown status, wait and retry
            await asyncio.sleep(10)
            response = client.responses.retrieve(response.id)
    
    if response.status == "failed":
        error_msg = getattr(response, "error", "Unknown error")
        raise RuntimeError(f"Deep Research task failed: {error_msg}")
    elif response.status == "cancelled":
        raise RuntimeError("Deep Research task was cancelled")
    
    # Extract output_text from response
    output_text = response.output_text
    
    # Try to extract JSON from the output text
    # The model might return JSON wrapped in markdown code blocks or plain text
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # If no JSON found, try parsing the whole output_text
            json_str = output_text
    
    # Parse JSON response
    try:
        response_data = json.loads(json_str)
    except json.JSONDecodeError:
        # If parsing fails, try to extract just the sections part
        # This is a fallback in case the output format is different
        raise ValueError(f"Failed to parse JSON from Deep Research output: {output_text[:500]}")
    
    # Normalize citation IDs to strings (they may come as integers from the model)
    if "sections" in response_data:
        for section in response_data["sections"]:
            if "citations" in section:
                for citation in section["citations"]:
                    if "id" in citation and isinstance(citation["id"], int):
                        citation["id"] = str(citation["id"])
    
    return SQAResponse.model_validate(response_data)


@solver
def openai_dr_solver(model: str = "o3-deep-research") -> Solver:
    """Solver using OpenAI Deep Research API.
    
    Args:
        model: Deep research model name (default: "o3-deep-research")
               Options: "o3-deep-research" or "o4-mini-deep-research"
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = state.metadata["initial_prompt"]
        for attempt in range(2):
            try:
                response = await query_openai_deep_research(model, question)
                content = json.dumps({"response": response.model_dump(mode="json")}, indent=2)
                state.messages.append(ChatMessageAssistant(content=content))
                state.output.completion = content
                return state
            except ValidationError:
                if attempt == 0:
                    await asyncio.sleep(2)
                    continue
                raise
        return state

    return solve
