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


def _generate_example_from_model(model_class: Type[BaseModel]) -> dict:
    """
    Generate a concrete example from a Pydantic model for use in prompts.

    Args:
        model_class: Pydantic model class

    Returns:
        Dictionary with example values based on field descriptions and examples
    """
    example = {}
    schema = model_class.model_json_schema()

    if "properties" in schema:
        for field_name, field_info in schema["properties"].items():
            # Check for example in json_schema_extra first
            if (
                "json_schema_extra" in field_info
                and "example" in field_info["json_schema_extra"]
            ):
                example[field_name] = field_info["json_schema_extra"]["example"]
            # Check for example at field level
            elif "example" in field_info:
                example[field_name] = field_info["example"]
            # Generate based on type
            elif field_info.get("type") == "string":
                example[field_name] = f"example_{field_name}"
            elif field_info.get("type") == "integer":
                example[field_name] = 0
            elif field_info.get("type") == "number":
                example[field_name] = 0.0
            elif field_info.get("type") == "boolean":
                example[field_name] = True
            elif field_info.get("type") == "array":
                items = field_info.get("items", {})
                if items.get("type") == "string":
                    example[field_name] = ["example_item_1", "example_item_2"]
                elif items.get("type") == "integer":
                    example[field_name] = [0, 1]
                else:
                    example[field_name] = []
            elif field_info.get("type") == "object":
                example[field_name] = {}
            else:
                example[field_name] = None

    return example


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
    # Generate example from model
    example = _generate_example_from_model(model_class)
    example_str = json.dumps(example, indent=2)

    enhanced_template = f"""{prompt_template}

IMPORTANT: You must respond with valid JSON data, NOT a schema description.

Return the actual data in this exact format:
{{example}}

The JSON schema is:
{{json_schema}}

Return ONLY the JSON data matching the example format above. Do not include any text before or after the JSON."""

    # Get the JSON schema from the Pydantic model
    schema = model_class.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    # Escape curly braces in schema and example to prevent LangChain from interpreting them as template variables
    schema_str = schema_str.replace("{", "{{").replace("}", "}}")
    example_str = example_str.replace("{", "{{").replace("}", "}}")
    enhanced_template = enhanced_template.replace("{example}", example_str)
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

            # Log parsed data before validation for debugging
            logger.debug(
                f"Parsed JSON for {model_class.__name__}: {json.dumps(data, indent=2)}"
            )

            # Validate against Pydantic model
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                f"JSON parsing/validation failed for {model_class.__name__}: {e}"
            )
            logger.error(f"Raw LLM response (first 500 chars): {response[:500]}")
            # Try to log what was parsed if JSON parsing succeeded but validation failed
            try:
                if response.startswith("{"):
                    parsed = json.loads(response)
                else:
                    start_idx = response.find("{")
                    end_idx = response.rfind("}") + 1
                    if start_idx != -1 and end_idx > 0:
                        parsed = json.loads(response[start_idx:end_idx])
                        logger.error(
                            f"Parsed JSON (before validation): {json.dumps(parsed, indent=2)}"
                        )
            except Exception as parse_error:
                logger.error(f"Could not parse JSON for debugging: {parse_error}")
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
    # Generate example from model
    example = _generate_example_from_model(model_class)
    example_str = json.dumps(example, indent=2)

    enhanced_template = f"""{prompt_template}

IMPORTANT: You must respond with valid JSON data, NOT a schema description.

Return the actual data in this exact format:
{{example}}

The JSON schema is:
{{json_schema}}

Return ONLY the JSON data matching the example format above. Do not include any text before or after the JSON."""

    schema = model_class.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    # Escape curly braces in schema and example to prevent LangChain from interpreting them as template variables
    schema_str = schema_str.replace("{", "{{").replace("}", "}}")
    example_str = example_str.replace("{", "{{").replace("}", "}}")
    enhanced_template = enhanced_template.replace("{example}", example_str)
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

            # Log parsed data before validation for debugging
            logger.debug(
                f"Parsed JSON for {model_class.__name__}: {json.dumps(data, indent=2)}"
            )

            # Validate against Pydantic model
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                f"JSON parsing/validation failed for {model_class.__name__}: {e}"
            )
            logger.error(f"Raw LLM response (first 500 chars): {response[:500]}")
            # Try to log what was parsed if JSON parsing succeeded but validation failed
            try:
                if response.startswith("{"):
                    parsed = json.loads(response)
                else:
                    start_idx = response.find("{")
                    end_idx = response.rfind("}") + 1
                    if start_idx != -1 and end_idx > 0:
                        parsed = json.loads(response[start_idx:end_idx])
                        logger.error(
                            f"Parsed JSON (before validation): {json.dumps(parsed, indent=2)}"
                        )
            except Exception as parse_error:
                logger.error(f"Could not parse JSON for debugging: {parse_error}")
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
