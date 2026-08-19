# -*- coding: utf-8 -*-

"""
Converters for transforming OpenAI Responses API format to Kiro format.

Converts Responses API input (string or item array) to the unified format
used by converters_core.py.
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from kiro.config import HIDDEN_MODELS
from kiro.model_resolver import get_model_id_for_kiro
from kiro.models_responses import CreateResponseRequest

from kiro.converters_core import (
    extract_text_content,
    extract_images_from_content,
    UnifiedMessage,
    UnifiedTool,
    ThinkingConfig,
    build_kiro_payload as core_build_kiro_payload,
)
from kiro.converters_openai import reasoning_effort_to_budget


# ==================================================================================================
# Input conversion
# ==================================================================================================

def _content_to_text(content: Any) -> str:
    """Extract plain text from a Responses API content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("type", "")
                if t in ("input_text", "output_text", "text"):
                    parts.append(block.get("text", ""))
                elif t == "refusal":
                    parts.append(block.get("refusal", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _extract_images_from_response_content(content: Any) -> List[Dict[str, Any]]:
    """Extract images from a Responses API content value."""
    if not isinstance(content, list):
        return []
    openai_style = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "input_image":
            url = block.get("image_url", "")
            if url:
                openai_style.append({"type": "image_url", "image_url": {"url": url}})
    return extract_images_from_content(openai_style)


def convert_responses_input_to_unified(
    input_value: Any,
    instructions: Optional[str],
) -> Tuple[str, List[UnifiedMessage]]:
    """
    Convert Responses API input to (system_prompt, unified_messages).

    input_value can be:
    - str: treated as a single user message
    - list of items: message / function_call / function_call_output objects
    """
    system_prompt = instructions or ""

    # Plain string input → single user message
    if isinstance(input_value, str):
        return system_prompt, [UnifiedMessage(role="user", content=input_value)]

    if not isinstance(input_value, list):
        return system_prompt, []

    unified: List[UnifiedMessage] = []
    # Accumulates function_call_output items until the next user message claims them.
    # Flushing as a standalone empty-user message would create two consecutive user
    # messages that merge_adjacent_messages collapses incorrectly for multi-turn threads.
    pending_tool_results: List[Dict[str, Any]] = []

    # Item types produced by built-in hosted tools that Kiro doesn't support.
    # Defined once outside the loop.
    _SKIP_ITEM_TYPES = frozenset({
        "reasoning",
        "computer_call",
        "computer_call_output",
        "web_search_call",
        "web_search_call_output",
        "image_generation_call",
        "image_generation_call_output",
        "code_interpreter_call",
        "code_interpreter_call_output",
        "file_search_call",
        "file_search_call_output",
        "mcp_call",
        "mcp_call_output",
    })

    for item in input_value:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "message")
        role = item.get("role", "")

        if item_type in _SKIP_ITEM_TYPES:
            logger.debug(f"Skipping unsupported Responses input item type '{item_type}'")
            continue

        if item_type == "function_call_output":
            # Accumulate tool results; they will be attached to the next user message.
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": item.get("call_id", ""),
                "content": item.get("output", "") or "(empty result)",
            })
            continue

        if item_type == "function_call":
            # Pending tool results before a new function_call mean a multi-step tool
            # chain where no user message arrived between calls.  Attach them to a
            # synthetic user message so Kiro history stays valid.
            if pending_tool_results:
                unified.append(UnifiedMessage(
                    role="user",
                    content="",
                    tool_results=list(pending_tool_results),
                ))
                pending_tool_results.clear()

            tool_calls = [{
                "id": item.get("id") or item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            }]
            unified.append(UnifiedMessage(
                role="assistant",
                content="",
                tool_calls=tool_calls,
            ))
            continue

        # message item
        if item_type == "message" or role in ("user", "assistant", "system"):
            content_raw = item.get("content", "")

            if role == "system":
                # Pending tool results before a system message: flush as standalone.
                if pending_tool_results:
                    unified.append(UnifiedMessage(
                        role="user",
                        content="",
                        tool_results=list(pending_tool_results),
                    ))
                    pending_tool_results.clear()
                system_prompt = (system_prompt + "\n" + _content_to_text(content_raw)).strip()
                continue

            text = _content_to_text(content_raw)
            images = _extract_images_from_response_content(content_raw) or None

            # Check for output items inside assistant messages (e.g. nested function_call)
            if role == "assistant" and isinstance(content_raw, list):
                tool_calls = []
                for block in content_raw:
                    if isinstance(block, dict) and block.get("type") == "function_call":
                        tool_calls.append({
                            "id": block.get("id") or block.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": block.get("arguments", "{}"),
                            },
                        })
                if tool_calls:
                    unified.append(UnifiedMessage(
                        role="assistant",
                        content=text,
                        tool_calls=tool_calls,
                        images=images,
                    ))
                    continue

            if role == "user" and pending_tool_results:
                # Attach accumulated tool results directly to this user message
                # instead of creating a separate empty-user message first.
                unified.append(UnifiedMessage(
                    role="user",
                    content=text,
                    tool_results=list(pending_tool_results),
                    images=images,
                ))
                pending_tool_results.clear()
                continue

            unified.append(UnifiedMessage(
                role=role or "user",
                content=text,
                images=images,
            ))

    # Flush any remaining tool results (e.g. conversation ended after tool call).
    if pending_tool_results:
        unified.append(UnifiedMessage(
            role="user",
            content="",
            tool_results=list(pending_tool_results),
        ))

    return system_prompt, unified


def convert_responses_tools_to_unified(tools: Optional[List[Any]]) -> Optional[List[UnifiedTool]]:
    """Convert Responses API tool definitions to unified format."""
    if not tools:
        return None

    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        t = tool.get("type", "function")
        if t != "function":
            # Built-in hosted tools (web_search_preview, computer_use_preview, etc.)
            # are not supported by Kiro — skip them silently.
            logger.debug(f"Skipping non-function tool type '{t}' (not supported by Kiro)")
            continue
        result.append(UnifiedTool(
            name=tool.get("name", ""),
            description=tool.get("description"),
            input_schema=tool.get("parameters"),
        ))

    return result or None


def extract_thinking_config_from_responses(request: CreateResponseRequest) -> ThinkingConfig:
    """Extract ThinkingConfig from a Responses API request."""
    if not request.reasoning_effort:
        return ThinkingConfig(enabled=True, budget_tokens=None)

    if request.reasoning_effort == "none":
        return ThinkingConfig(enabled=False, budget_tokens=None)

    max_tokens = request.max_output_tokens or 4096
    budget = reasoning_effort_to_budget(max_tokens, request.reasoning_effort)
    return ThinkingConfig(enabled=True, budget_tokens=budget)


def build_kiro_payload_from_responses(
    request_data: CreateResponseRequest,
    conversation_id: str,
    profile_arn: str,
) -> dict:
    """Build Kiro API payload from a Responses API request."""
    import json as _json

    # Debug: log the raw incoming Responses request fields relevant to conversion
    if logger.level("DEBUG").no >= 0:
        raw_tools = request_data.tools or []
        tool_summary = [
            {"type": t.get("type", "?"), "name": t.get("name", "")}
            for t in raw_tools if isinstance(t, dict)
        ]
        raw_input = request_data.input
        input_types = (
            [i.get("type", i.get("role", "?")) for i in raw_input if isinstance(i, dict)]
            if isinstance(raw_input, list) else type(raw_input).__name__
        )
        logger.debug(
            f"[Responses] incoming: model={request_data.model}, "
            f"input_item_types={input_types}, tools={tool_summary}"
        )

    system_prompt, unified_messages = convert_responses_input_to_unified(
        request_data.input, request_data.instructions
    )
    unified_tools = convert_responses_tools_to_unified(request_data.tools)
    model_id = get_model_id_for_kiro(request_data.model, HIDDEN_MODELS)
    thinking_config = extract_thinking_config_from_responses(request_data)

    logger.debug(
        f"[Responses] converting: model={request_data.model} -> {model_id}, "
        f"unified_messages={len(unified_messages)}, unified_tools={len(unified_tools) if unified_tools else 0}"
    )

    result = core_build_kiro_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=model_id,
        tools=unified_tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        thinking_config=thinking_config,
    )

    # Debug: log the generated Kiro payload (truncated for readability)
    try:
        payload_str = _json.dumps(result.payload, ensure_ascii=False)
        logger.debug(f"[Responses] kiro payload ({len(payload_str)} bytes): {payload_str[:2000]}")
    except Exception:
        pass

    return result.payload
