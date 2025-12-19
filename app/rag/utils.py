"""
Utility functions for RAG techniques.

Includes helpers for structured output, LLM chains, and other common operations.
"""

import json
import logging
from typing import Any, Type, TypeVar

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def build_structured_chain_with_fallback(
    llm: BaseLanguageModel,
    prompt_template: str,
    model_class: Type[T],
) -> Any:
    """
    Build a chain that returns structured output with fallback support.

    Attempts to use `with_structured_output()` first (for LLMs that support it).
    Falls back to JSON parsing approach when `NotImplementedError` is raised.

    Args:
        llm: Language model instance
        prompt_template: Prompt template string with variables in {brackets}
        model_class: Pydantic model class for structured output

    Returns:
        Chain that returns structured output (BaseModel instance)
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Try native structured output first
    try:
        structured_llm = llm.with_structured_output(model_class)
        logger.debug(f"Using native structured output for {model_class.__name__}")
        return prompt | structured_llm
    except NotImplementedError:
        logger.debug(
            f"Native structured output not supported, using JSON parsing fallback for {model_class.__name__}"
        )

    # Fallback: Use JSON parsing approach
    # Create an enhanced prompt that asks for JSON output
    enhanced_template = f"""{prompt_template}

Respond ONLY with valid JSON that matches this schema:
{{json_schema}}

Do not include any text before or after the JSON."""

    # Get the JSON schema from the Pydantic model
    schema = model_class.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    enhanced_template = enhanced_template.replace("{json_schema}", schema_str)

    # Build parser that handles both raw JSON and strings containing JSON
    parser = JsonOutputParser(pydantic_object=model_class)

    enhanced_prompt = ChatPromptTemplate.from_template(enhanced_template)

    # Create chain with LLM and custom parsing
    def parse_json_response(response: str) -> T:
        """Parse JSON response and validate against Pydantic model."""
        try:
            # Try to parse JSON directly
            if response.startswith("{"):
                data = json.loads(response)
            else:
                # Try to extract JSON from response
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON found in response")
                data = json.loads(response[start_idx:end_idx])

            # Validate against Pydantic model
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"JSON parsing/validation failed: {e}")
            logger.debug(f"Raw response: {response}")
            raise

    # Return a chain-like object that can be used with |
    class StructuredOutputChain:
        """Wrapper for structured output chain."""

        def __init__(self, prompt, llm, parser_func):
            self.prompt = prompt
            self.llm = llm
            self.parser = parser_func

        async def ainvoke(self, input_dict: dict, **kwargs) -> T:
            """Async invoke the chain."""
            # Invoke prompt with input
            prompt_value = await self.prompt.ainvoke(input_dict)
            # Call LLM
            response = await self.llm.ainvoke(prompt_value)
            # Get string content from response
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)
            # Parse JSON
            return self.parser(content)

        def __or__(self, other):
            """Support pipe operator for compatibility."""
            return self

    return StructuredOutputChain(enhanced_prompt, llm, parse_json_response)


def build_structured_chain(
    llm: BaseLanguageModel,
    prompt_template: str,
    model_class: Type[T],
) -> Any:
    """
    Synchronous wrapper for building structured output chains.

    Uses LangChain's built-in structured output if available,
    otherwise falls back to JSON parsing.

    Args:
        llm: Language model instance
        prompt_template: Prompt template string
        model_class: Pydantic model class

    Returns:
        Chain that returns structured output
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Try native structured output first
    try:
        structured_llm = llm.with_structured_output(model_class)
        logger.debug(f"Using native structured output for {model_class.__name__}")
        return prompt | structured_llm
    except NotImplementedError:
        logger.debug(
            f"Native structured output not supported, using JSON parsing fallback for {model_class.__name__}"
        )

    # Fallback: Use JSON parsing approach
    enhanced_template = f"""{prompt_template}

Respond ONLY with valid JSON that matches this schema:
{{json_schema}}

Do not include any text before or after the JSON."""

    schema = model_class.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    enhanced_template = enhanced_template.replace("{json_schema}", schema_str)

    enhanced_prompt = ChatPromptTemplate.from_template(enhanced_template)
    parser = JsonOutputParser(pydantic_object=model_class)

    def parse_json_response(response: str) -> T:
        """Parse JSON response and validate against Pydantic model."""
        try:
            # Try to parse JSON directly
            if response.startswith("{"):
                data = json.loads(response)
            else:
                # Try to extract JSON from response
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON found in response")
                data = json.loads(response[start_idx:end_idx])

            # Validate against Pydantic model
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"JSON parsing/validation failed: {e}")
            logger.debug(f"Raw response: {response}")
            raise

    class StructuredOutputChain:
        """Wrapper for structured output chain."""

        def __init__(self, prompt, llm, parser_func):
            self.prompt = prompt
            self.llm = llm
            self.parser = parser_func

        def invoke(self, input_dict: dict, **kwargs) -> T:
            """Invoke the chain synchronously."""
            prompt_value = self.prompt.invoke(input_dict)
            response = self.llm.invoke(prompt_value)
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)
            return self.parser(content)

        async def ainvoke(self, input_dict: dict, **kwargs) -> T:
            """Async invoke the chain."""
            prompt_value = await self.prompt.ainvoke(input_dict)
            response = await self.llm.ainvoke(prompt_value)
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)
            return self.parser(content)

        def __or__(self, other):
            """Support pipe operator for compatibility."""
            return self

    return StructuredOutputChain(enhanced_prompt, llm, parse_json_response)
