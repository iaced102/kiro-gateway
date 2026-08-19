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
    dropped_tool_names: Optional[set] = None,
) -> Tuple[str, List[UnifiedMessage]]:
    """
    Convert Responses API input to (system_prompt, unified_messages).

    input_value can be:
    - str: treated as a single user message
    - list of items: message / function_call / function_call_output objects

    dropped_tool_names: set of tool names whose definitions were skipped (e.g. hosted
    tools like web_search).  Any function_call item whose name is in this set is
    silently dropped together with its matching function_call_output items, so that
    Kiro history never references a tool that was not forwarded as a toolSpecification.
    """
    system_prompt = instructions or ""
    _dropped = dropped_tool_names or set()

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

    # call_ids whose function_call was dropped because the tool definition was skipped.
    # The matching function_call_output items must also be dropped to keep history
    # consistent (Kiro validates tool-use ↔ tool-result pairing).
    dropped_call_ids: set = set()

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
            call_id = item.get("call_id", "")
            if call_id in dropped_call_ids:
                logger.debug(
                    f"Dropping function_call_output call_id='{call_id}' "
                    f"(matching function_call was dropped)"
                )
                continue
            # Accumulate tool results; they will be attached to the next user message.
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": item.get("output", "") or "(empty result)",
            })
            continue

        if item_type == "function_call":
            logger.debug(
                f"[Responses] RAW function_call: "
                f"id={item.get('id')!r} "
                f"call_id={item.get('call_id')!r} "
                f"name={item.get('name')!r} "
                f"namespace={item.get('namespace')!r} "
                f"keys={list(item.keys())!r}"
            )
            fc_name = item.get("name", "")
            fc_id = item.get("call_id") or item.get("id") or ""

            if not fc_name:
                # Historical function_call with empty name — produced by earlier gateway
                # bug where tool name/args were serialized from the wrong dict level.
                # Drop the structured call and its matching output to avoid generating
                # a Kiro toolUse with name="" which causes REQUEST_BODY_INVALID.
                logger.warning(
                    f"[Responses] Dropping function_call with empty name "
                    f"(id={item.get('id')!r} call_id={item.get('call_id')!r}). "
                    f"Matching function_call_output will also be dropped."
                )
                dropped_call_ids.add(fc_id)
                continue

            if fc_name in _dropped:
                # The tool definition was skipped (e.g. web_search hosted tool).
                # Record the call_id so the matching function_call_output is also
                # dropped — an orphaned tool result would cause REQUEST_BODY_INVALID.
                dropped_call_ids.add(fc_id)
                logger.debug(
                    f"Dropping function_call name='{fc_name}' id='{fc_id}' "
                    f"(tool definition was skipped)"
                )
                continue

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
                "id": fc_id,
                "type": "function",
                "function": {
                    "name": fc_name,
                    "arguments": item.get("arguments", "{}"),
                },
            }]
            logger.debug(f"[Responses] function_call -> tool_calls={tool_calls!r}")
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
                            "id": block.get("call_id") or block.get("id") or "",
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


def convert_responses_tools_to_unified(
    tools: Optional[List[Any]],
) -> tuple:
    """Convert Responses API tool definitions to unified format.

    Returns (unified_tools_or_None, dropped_tool_names) where dropped_tool_names
    is the set of tool names whose definitions were skipped so callers can strip
    any history items that reference them.
    """
    if not tools:
        return None, set()

    result = []
    dropped_names: set = set()

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        t = tool.get("type", "function")
        name = tool.get("name", "")

        if t == "function":
            result.append(UnifiedTool(
                name=name,
                description=tool.get("description"),
                input_schema=tool.get("parameters"),
            ))
        elif t == "namespace":
            # Codex uses "namespace" as a callable tool group (e.g. the local shell
            # namespace).  Map it to a plain function tool so that any function_call
            # items referencing it remain valid in Kiro history.
            params = (
                tool.get("parameters")
                or tool.get("input_schema")
                or tool.get("schema")
            )
            result.append(UnifiedTool(
                name=name,
                description=tool.get("description") or f"Namespace: {name}",
                input_schema=params,
            ))
            logger.debug(f"Mapped namespace tool '{name}' to function tool")
        else:
            # Hosted tools (web_search, computer_use_preview, …) are not supported
            # by Kiro.  Record the name so the input converter can drop any
            # function_call / function_call_output items that reference it.
            logger.debug(
                f"Skipping unsupported tool type '{t}' (name='{name}')"
            )
            if name:
                dropped_names.add(name)

    return result or None, dropped_names


def _log_kiro_payload_summary(payload: dict) -> None:
    """Emit a compact DEBUG summary of the Kiro payload about to be sent."""
    try:
        conv = payload.get("conversationState", {})
        history = conv.get("history", [])
        current = conv.get("currentMessage", {})
        uim = current.get("userInputMessage", {})
        ctx = uim.get("userInputMessageContext", {})
        available_tools = [
            t.get("toolSpecification", {}).get("name", "?")
            for t in ctx.get("tools", [])
        ]

        lines = ["[Responses] kiro payload summary:", "history:"]
        for i, entry in enumerate(history):
            if "userInputMessage" in entry:
                m = entry["userInputMessage"]
                tr_ids = [
                    r.get("toolUseId", "?")
                    for r in m.get("userInputMessageContext", {}).get("toolResults", [])
                ]
                suffix = f" tool_results={tr_ids}" if tr_ids else ""
                lines.append(f"  {i} user{suffix}")
            elif "assistantResponseMessage" in entry:
                m = entry["assistantResponseMessage"]
                tu_names = [
                    f"{u.get('name', '?')}:{u.get('toolUseId', '?')}"
                    for u in m.get("toolUses", [])
                ]
                suffix = f" tool_use={tu_names}" if tu_names else ""
                lines.append(f"  {i} assistant{suffix}")

        cur_tr_ids = [
            r.get("toolUseId", "?")
            for r in ctx.get("toolResults", [])
        ]
        suffix = f" tool_results={cur_tr_ids}" if cur_tr_ids else ""
        lines.append(f"  {len(history)} user (current){suffix}")

        if available_tools:
            lines.append("available_tools:")
            for name in available_tools:
                lines.append(f"  - {name}")
        else:
            lines.append("available_tools: (none)")

        logger.debug("\n".join(lines))
    except Exception:
        pass


def _validate_kiro_tool_consistency(payload: dict) -> None:
    """
    Validate tool use / tool result consistency in the Kiro payload.

    Logs a DEBUG line per tool interaction showing name, IDs, and MATCH status.
    Raises ValueError (→ HTTP 400) when:
    - any toolUse has an empty name
    - any toolResult has no matching toolUse in the immediately preceding assistant entry
    """
    conv = payload.get("conversationState", {})
    history = conv.get("history", [])
    current = conv.get("currentMessage", {})
    uim = current.get("userInputMessage", {})
    cur_ctx = uim.get("userInputMessageContext", {})

    # Build a flat list of (entry_type, data) for sequential scanning
    entries = list(history) + [{"userInputMessage": uim}]

    last_tool_use_ids: set = set()
    errors = []

    for entry in entries:
        if "assistantResponseMessage" in entry:
            arm = entry["assistantResponseMessage"]
            last_tool_use_ids = set()
            for tu in arm.get("toolUses", []):
                name = tu.get("name", "")
                uid = tu.get("toolUseId", "")
                last_tool_use_ids.add(uid)
                logger.debug(
                    f"[Responses] tool_use name={name!r} id={uid!r}"
                )
                if not name:
                    errors.append(f"toolUse has empty name (toolUseId={uid!r})")

        elif "userInputMessage" in entry:
            m = entry["userInputMessage"]
            ctx = m.get("userInputMessageContext", {})
            for tr in ctx.get("toolResults", []):
                uid = tr.get("toolUseId", "")
                match = uid in last_tool_use_ids
                logger.debug(
                    f"[Responses] tool_result id={uid!r} MATCH={match}"
                )
                if not match and uid:
                    errors.append(
                        f"toolResult toolUseId={uid!r} has no matching toolUse "
                        f"(preceding toolUse IDs: {sorted(last_tool_use_ids)})"
                    )

    if errors:
        raise ValueError(
            "Kiro payload tool consistency error: " + "; ".join(errors)
        )


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

    # Convert tools first so we know which names were dropped before processing input.
    unified_tools, dropped_tool_names = convert_responses_tools_to_unified(request_data.tools)

    system_prompt, unified_messages = convert_responses_input_to_unified(
        request_data.input, request_data.instructions, dropped_tool_names
    )
    model_id = get_model_id_for_kiro(request_data.model, HIDDEN_MODELS)
    thinking_config = extract_thinking_config_from_responses(request_data)

    logger.debug(
        f"[Responses] converting: model={request_data.model} -> {model_id}, "
        f"unified_messages={len(unified_messages)}, unified_tools={len(unified_tools) if unified_tools else 0}, "
        f"dropped_tools={sorted(dropped_tool_names) if dropped_tool_names else []}"
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

    # Compact DEBUG summary of what will be sent to Kiro
    _log_kiro_payload_summary(result.payload)

    # Validate tool use / result consistency — raises ValueError (→ HTTP 400) on mismatch
    _validate_kiro_tool_consistency(result.payload)

    return result.payload
