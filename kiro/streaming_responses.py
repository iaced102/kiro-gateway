# -*- coding: utf-8 -*-

"""
Streaming logic for converting Kiro stream to OpenAI Responses API format.

SSE events emitted follow the OpenAI Responses API streaming spec:
  response.created, response.in_progress,
  response.output_item.added, response.content_part.added,
  response.output_text.delta, response.output_text.done,
  response.content_part.done, response.output_item.done,
  response.function_call_arguments.delta,
  response.function_call_arguments.done,
  response.completed  ← terminal event (NOT response.done)
"""

import json
import time
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Awaitable, Optional, List, Dict, Any

import httpx
from loguru import logger

from kiro.utils import generate_completion_id
from kiro.config import FIRST_TOKEN_TIMEOUT, FIRST_TOKEN_MAX_RETRIES
from kiro.tokenizer import count_tokens, count_message_tokens, count_tools_tokens

from kiro.streaming_core import (
    parse_kiro_stream,
    FirstTokenTimeoutError,
    KiroEvent,
    calculate_tokens_from_context_usage,
    stream_with_first_token_retry as stream_with_first_token_retry_core,
)

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager
    from kiro.cache import ModelInfoCache

try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


def _normalize_tool_use(tool: dict) -> tuple:
    """Extract (call_id, name, arguments) from a KiroEvent tool_use dict.

    KiroEvent.tool_use uses the gateway's unified/OpenAI-style format:
      {"id": "tooluse_...", "type": "function", "function": {"name": "...", "arguments": "..."}}

    This is the internal contract established by AwsEventStreamParser._process_tool_start_event.
    """
    call_id = tool.get("id", "")
    func = tool.get("function") or {}
    name = func.get("name", "")
    arguments = func.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    if not name:
        logger.error(f"[Responses] tool_use has empty name: raw={tool!r}")
    return call_id, name, arguments



    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_kiro_to_responses(
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert a Kiro API streaming response to OpenAI Responses API SSE format.
    """
    response_id = f"resp_{generate_completion_id()}"
    message_item_id = f"msg_{generate_completion_id()}"
    created_at = int(time.time())

    # Accumulated state
    text_buffer = ""
    tool_items: List[Dict[str, Any]] = []  # {id, call_id, name, arguments, output_index}
    current_tool_index: Optional[int] = None  # output_index of the active tool item
    content_started = False  # whether we've emitted the text content part

    input_tokens = 0
    output_tokens = 0

    # Emit response.created
    initial_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "output": [],
        "status": "in_progress",
        "usage": None,
    }
    yield _sse("response.created", {"type": "response.created", "response": initial_response})
    yield _sse("response.in_progress", {"type": "response.in_progress", "response": initial_response})

    try:
        async for event in parse_kiro_stream(
            response,
            first_token_timeout=first_token_timeout,
        ):
            if event.type == "content":
                # If we haven't started the text output item yet, emit it now
                if not content_started:
                    content_started = True
                    output_index = len(tool_items)  # text comes after any leading tool calls
                    # output_item.added for the message
                    yield _sse("response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "type": "message",
                            "id": message_item_id,
                            "role": "assistant",
                            "content": [],
                            "status": "in_progress",
                        },
                    })
                    # content_part.added
                    yield _sse("response.content_part.added", {
                        "type": "response.content_part.added",
                        "item_id": message_item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": ""},
                    })

                text_buffer += event.content or ""
                yield _sse("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": len(tool_items),
                    "content_index": 0,
                    "delta": event.content or "",
                })

            elif event.type == "tool_use":
                tool = event.tool_use or {}
                tool_output_index = len(tool_items)
                call_id, name, arguments = _normalize_tool_use(tool)
                tool_item = {
                    "id": f"fc_{call_id}",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "output_index": tool_output_index,
                }
                tool_items.append(tool_item)
                current_tool_index = tool_output_index

                logger.debug(
                    f"[Responses Streaming] emitting function_call "
                    f"name={name!r} id='fc_{call_id}' call_id={call_id!r} "
                    f"arguments_length={len(arguments)}"
                )

                yield _sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": tool_output_index,
                    "item": {
                        "type": "function_call",
                        "id": tool_item["id"],
                        "call_id": tool_item["call_id"],
                        "name": tool_item["name"],
                        "arguments": "",
                        "status": "in_progress",
                    },
                })
                # Emit arguments as a single delta + done
                yield _sse("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": tool_item["id"],
                    "output_index": tool_output_index,
                    "delta": tool_item["arguments"],
                })
                yield _sse("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "item_id": tool_item["id"],
                    "output_index": tool_output_index,
                    "arguments": tool_item["arguments"],
                })
                yield _sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": tool_output_index,
                    "item": {
                        "type": "function_call",
                        "id": tool_item["id"],
                        "call_id": tool_item["call_id"],
                        "name": tool_item["name"],
                        "arguments": tool_item["arguments"],
                        "status": "completed",
                    },
                })

            elif event.type == "usage":
                usage = event.usage or {}
                # Try to extract token counts from usage event
                input_tokens = usage.get("input_tokens", input_tokens)
                output_tokens = usage.get("output_tokens", output_tokens)

            elif event.type == "context_usage":
                if event.context_usage_percentage and model_cache:
                    try:
                        tokens = await calculate_tokens_from_context_usage(
                            event.context_usage_percentage,
                            model,
                            model_cache,
                            auth_manager,
                        )
                        if tokens and input_tokens == 0:
                            input_tokens = tokens
                    except Exception:
                        pass

            elif event.type == "error":
                logger.warning(f"Kiro stream error event: {event}")

    except FirstTokenTimeoutError:
        raise
    except Exception as e:
        logger.error(f"Error in stream_kiro_to_responses: {e}", exc_info=True)
        raise

    # Finalize text output item
    text_output_index = len(tool_items)
    if content_started:
        yield _sse("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": message_item_id,
            "output_index": text_output_index,
            "content_index": 0,
            "text": text_buffer,
        })
        yield _sse("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": message_item_id,
            "output_index": text_output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": text_buffer},
        })
        yield _sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": text_output_index,
            "item": {
                "type": "message",
                "id": message_item_id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": text_buffer, "annotations": []}],
                "status": "completed",
            },
        })

    # Fallback token estimation
    if input_tokens == 0 and request_messages:
        try:
            input_tokens = count_message_tokens(request_messages)
            if request_tools:
                input_tokens += count_tools_tokens(request_tools)
        except Exception:
            pass
    if output_tokens == 0 and text_buffer:
        try:
            output_tokens = count_tokens(text_buffer)
        except Exception:
            pass

    # Build final output list
    output_list = []
    for t in tool_items:
        output_list.append({
            "type": "function_call",
            "id": t["id"],
            "call_id": t["call_id"],
            "name": t["name"],
            "arguments": t["arguments"],
            "status": "completed",
        })
    if content_started:
        output_list.append({
            "type": "message",
            "id": message_item_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": text_buffer, "annotations": []}],
            "status": "completed",
        })

    final_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "output": output_list,
        "status": "completed",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    yield _sse("response.completed", {"type": "response.completed", "response": final_response})


async def collect_responses_response(
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> dict:
    """
    Collect a full (non-streaming) response in Responses API format.
    """
    response_id = f"resp_{generate_completion_id()}"
    message_item_id = f"msg_{generate_completion_id()}"
    created_at = int(time.time())

    text_buffer = ""
    tool_items: List[Dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0

    try:
        async for event in parse_kiro_stream(response):
            if event.type == "content":
                text_buffer += event.content or ""
            elif event.type == "tool_use":
                tool = event.tool_use or {}
                call_id, name, arguments = _normalize_tool_use(tool)
                tool_items.append({
                    "type": "function_call",
                    "id": f"fc_{call_id}",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed",
                })
                logger.debug(f"[Responses] collected function_call name={name!r} call_id={call_id!r}")
            elif event.type == "usage":
                usage = event.usage or {}
                input_tokens = usage.get("input_tokens", input_tokens)
                output_tokens = usage.get("output_tokens", output_tokens)
            elif event.type == "context_usage":
                if event.context_usage_percentage and model_cache:
                    try:
                        tokens = await calculate_tokens_from_context_usage(
                            event.context_usage_percentage,
                            model,
                            model_cache,
                            auth_manager,
                        )
                        if tokens and input_tokens == 0:
                            input_tokens = tokens
                    except Exception:
                        pass
    except Exception as e:
        logger.error(f"Error collecting responses response: {e}", exc_info=True)
        raise

    # Fallback token estimation
    if input_tokens == 0 and request_messages:
        try:
            input_tokens = count_message_tokens(request_messages)
            if request_tools:
                input_tokens += count_tools_tokens(request_tools)
        except Exception:
            pass
    if output_tokens == 0 and text_buffer:
        try:
            output_tokens = count_tokens(text_buffer)
        except Exception:
            pass

    output_list = list(tool_items)
    if text_buffer:
        output_list.append({
            "type": "message",
            "id": message_item_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": text_buffer, "annotations": []}],
            "status": "completed",
        })

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "output": output_list,
        "status": "completed",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


async def stream_with_first_token_retry_responses(
    make_request: Callable[[], Awaitable[httpx.Response]],
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    initial_response: httpx.Response,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """
    Wrapper that retries on FirstTokenTimeoutError, then delegates to stream_kiro_to_responses.
    """
    async def _do_stream(resp: httpx.Response) -> AsyncGenerator[str, None]:
        return stream_kiro_to_responses(
            resp, model, model_cache, auth_manager,
            request_messages=request_messages,
            request_tools=request_tools,
        )

    retries = 0
    current_response = initial_response

    while True:
        try:
            async for chunk in stream_kiro_to_responses(
                current_response, model, model_cache, auth_manager,
                request_messages=request_messages,
                request_tools=request_tools,
            ):
                yield chunk
            return
        except FirstTokenTimeoutError:
            retries += 1
            if retries > FIRST_TOKEN_MAX_RETRIES:
                logger.error(f"First token timeout after {retries} retries, giving up")
                raise
            logger.warning(f"First token timeout, retrying ({retries}/{FIRST_TOKEN_MAX_RETRIES})")
            current_response = await make_request()
