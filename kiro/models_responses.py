# -*- coding: utf-8 -*-

"""
Pydantic models for OpenAI Responses API.

Reference: https://platform.openai.com/docs/api-reference/responses
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ==================================================================================================
# Input models
# ==================================================================================================

class CreateResponseRequest(BaseModel):
    """Request body for POST /v1/responses."""
    model: str
    # input can be a plain string or an array of items (messages, function_call, function_call_output)
    input: Union[str, List[Any]]
    stream: bool = False

    # System prompt equivalent
    instructions: Optional[str] = None

    # Generation parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None

    # Reasoning
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] = None

    # Tools
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Truncation / multi-turn (accepted, not used for routing)
    truncation: Optional[str] = None
    previous_response_id: Optional[str] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Output models
# ==================================================================================================

class ResponseUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: Optional[Dict[str, Any]] = None
    output_tokens_details: Optional[Dict[str, Any]] = None


class Response(BaseModel):
    """Full non-streaming response for /v1/responses."""
    id: str
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str
    output: List[Any] = Field(default_factory=list)
    status: str = "completed"
    usage: Optional[ResponseUsage] = None
    error: Optional[Any] = None

    model_config = {"extra": "allow"}
