import asyncio
import json
import os
import re
import time

from astabench.types.sqa import SQAResponse
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import Generate, Solver, TaskState, solver
from google import genai
from pydantic import ValidationError


async def query_gemini_deep_research(
    question: str,
    timeout: int = 3600,
):
    """Query Google Gemini Deep Research API using the Interactions API.
    
    Args:
        question: The research question to answer
        timeout: Timeout in seconds for the API call (default: 3600 for 1 hour)
    
    Returns:
        SQAResponse object with the parsed response
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = (
        "Please write a report responding to the following user query. Return your result as valid JSON with a single key `sections` which is a JSON list of sections, each having keys `title`, `text`, and `citations`. Each claim in the text should be supported by an inline citation. Each entry in `citations` should have a JSON list of `snippets` extracted from the reference document and an `id`, each of which appears exactly in the text. Each `id` should be an inline citation. Any additional information about the citation should go under `metadata`.\n\n"
        + question
    )
    
    # Create deep research interaction with background execution
    # Background execution is required for long-running research tasks
    interaction = client.interactions.create(
        input=prompt,
        agent="deep-research-pro-preview-12-2025",
        background=True,
    )
    
    # Poll for results until completion or failure
    start_time = time.time()
    while True:
        # Check timeout
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Deep Research task exceeded timeout of {timeout} seconds")
        
        # Get current interaction status
        interaction = client.interactions.get(interaction.id)
        
        if interaction.status == "completed":
            # Extract output text from the last output
            output_text = interaction.outputs[-1].text
            break
        elif interaction.status == "failed":
            error_msg = getattr(interaction, "error", "Unknown error")
            raise RuntimeError(f"Deep Research task failed: {error_msg}")
        elif interaction.status == "in_progress":
            # Wait before polling again
            await asyncio.sleep(10)
        else:
            # Unknown status, wait and retry
            await asyncio.sleep(10)
    
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
def gemini_dr_solver() -> Solver:
    """Solver using Google Gemini Deep Research API.
    
    The Deep Research agent uses the Interactions API and runs in the background.
    It automatically uses web search tools to research the question.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = state.metadata["initial_prompt"]
        for attempt in range(2):
            try:
                response = await query_gemini_deep_research(question)
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
