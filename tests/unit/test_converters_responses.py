# -*- coding: utf-8 -*-

"""
Regression tests for converters_responses.py — Codex 0.148 compatibility.

Covers the exact failure shapes that produced REQUEST_BODY_INVALID:
- namespace tool definitions mapped to function tools
- function_call items for dropped tools (e.g. web_search) skipped together with
  their function_call_output counterparts
- developer role messages handled without creating invalid history
- multi-turn tool calls
- normal chat and fresh hello
"""

import pytest

from kiro.converters_responses import (
    convert_responses_input_to_unified,
    convert_responses_tools_to_unified,
)
from kiro.converters_core import UnifiedTool


# ==================================================================================================
# Tool conversion
# ==================================================================================================

class TestConvertResponsesToolsToUnified:

    def test_function_tool_returned(self):
        tools = [{"type": "function", "name": "my_func", "description": "desc",
                  "parameters": {"type": "object", "properties": {}}}]
        result, dropped = convert_responses_tools_to_unified(tools)
        assert len(result) == 1
        assert result[0].name == "my_func"
        assert dropped == set()

    def test_namespace_tool_mapped_to_function(self):
        tools = [{"type": "namespace", "name": "shell", "description": "Local shell"}]
        result, dropped = convert_responses_tools_to_unified(tools)
        assert len(result) == 1
        assert result[0].name == "shell"
        assert dropped == set()

    def test_namespace_without_description_gets_placeholder(self):
        tools = [{"type": "namespace", "name": "shell"}]
        result, dropped = convert_responses_tools_to_unified(tools)
        assert result[0].description == "Namespace: shell"

    def test_web_search_dropped_and_name_recorded(self):
        tools = [{"type": "web_search", "name": "web_search"}]
        result, dropped = convert_responses_tools_to_unified(tools)
        assert result is None
        assert "web_search" in dropped

    def test_unsupported_tool_without_name_not_in_dropped(self):
        tools = [{"type": "computer_use_preview"}]
        result, dropped = convert_responses_tools_to_unified(tools)
        assert result is None
        assert dropped == set()  # no name -> nothing to track

    def test_mixed_tools(self):
        tools = [
            {"type": "function", "name": "get_weather"},
            {"type": "namespace", "name": "shell"},
            {"type": "web_search", "name": "web_search"},
        ]
        result, dropped = convert_responses_tools_to_unified(tools)
        assert len(result) == 2
        names = {t.name for t in result}
        assert names == {"get_weather", "shell"}
        assert dropped == {"web_search"}

    def test_empty_list(self):
        result, dropped = convert_responses_tools_to_unified([])
        assert result is None
        assert dropped == set()

    def test_none(self):
        result, dropped = convert_responses_tools_to_unified(None)
        assert result is None
        assert dropped == set()


# ==================================================================================================
# Input conversion - basic
# ==================================================================================================

class TestConvertResponsesInputBasic:

    def test_string_input(self):
        sys, msgs = convert_responses_input_to_unified("hello", None)
        assert sys == ""
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "hello"

    def test_user_message_item(self):
        inp = [{"type": "message", "role": "user", "content": "hi"}]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        assert len(msgs) == 1
        assert msgs[0].role == "user"

    def test_instructions_become_system_prompt(self):
        sys, msgs = convert_responses_input_to_unified("hi", "You are helpful.")
        assert sys == "You are helpful."

    def test_system_message_appended_to_instructions(self):
        inp = [
            {"type": "message", "role": "system", "content": "extra context"},
            {"type": "message", "role": "user", "content": "hello"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, "base")
        assert "extra context" in sys
        assert "base" in sys

    def test_developer_role_becomes_non_system(self):
        inp = [
            {"type": "message", "role": "developer", "content": "context"},
            {"type": "message", "role": "user", "content": "question"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        roles = [m.role for m in msgs]
        assert "developer" in roles  # normalized to user downstream in build_kiro_payload
        assert len(msgs) == 2


# ==================================================================================================
# Input conversion - function calls with no dropped tools
# ==================================================================================================

class TestConvertResponsesInputFunctionCall:

    def test_function_call_becomes_assistant_with_tool_calls(self):
        inp = [
            {"type": "function_call", "name": "get_weather", "id": "call_1",
             "arguments": '{"city": "Paris"}'},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        assert len(msgs) == 1
        assert msgs[0].role == "assistant"
        tc = msgs[0].tool_calls[0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["id"] == "call_1"

    def test_function_call_output_becomes_tool_result(self):
        inp = [
            {"type": "function_call", "name": "get_weather", "id": "call_1",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "Sunny"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        assert msgs[0].role == "assistant"
        user_with_result = msgs[1]
        assert user_with_result.role == "user"
        assert any(tr["tool_use_id"] == "call_1" for tr in user_with_result.tool_results)

    def test_multi_turn_tool_calls(self):
        inp = [
            {"type": "message", "role": "user", "content": "step 1"},
            {"type": "function_call", "name": "tool_a", "id": "c1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "result_a"},
            {"type": "message", "role": "user", "content": "step 2"},
            {"type": "function_call", "name": "tool_b", "id": "c2", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c2", "output": "result_b"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        roles = [m.role for m in msgs]
        assert roles[0] == "user"       # "step 1"
        assert roles[1] == "assistant"  # tool_a call
        # c1 output attached to "step 2" user message
        user_step2 = next(m for m in msgs if m.role == "user" and m.content == "step 2")
        assert any(tr["tool_use_id"] == "c1" for tr in (user_step2.tool_results or []))


# ==================================================================================================
# Dropped tools - the core Codex 0.148 regression
# ==================================================================================================

class TestDroppedToolsConsistency:
    """
    When a tool definition is skipped (e.g. web_search), any function_call and
    function_call_output referencing that tool must also be dropped.  If they were
    kept, Kiro would see a toolUse referencing a toolSpecification that was never
    sent -> REQUEST_BODY_INVALID.
    """

    def test_web_search_call_dropped_when_tool_skipped(self):
        dropped = {"web_search"}
        inp = [
            {"type": "function_call", "name": "web_search", "id": "ws1",
             "call_id": "ws1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "ws1", "output": "results"},
            {"type": "message", "role": "user", "content": "ok"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None, dropped)
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "ok"

    def test_namespace_function_call_kept_when_tool_kept(self):
        dropped: set = set()
        inp = [
            {"type": "function_call", "name": "shell", "id": "sh1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "sh1", "output": "ok"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None, dropped)
        assert msgs[0].role == "assistant"
        assert msgs[0].tool_calls[0]["function"]["name"] == "shell"
        assert msgs[1].role == "user"
        assert any(tr["tool_use_id"] == "sh1" for tr in msgs[1].tool_results)

    def test_mixed_namespace_and_web_search_history(self):
        """
        Codex 0.148 canonical shape: namespace + web_search tools defined,
        history contains both function_call types.
        Only the namespace function_call/output must survive.
        """
        dropped = {"web_search"}
        inp = [
            {"type": "function_call", "name": "shell", "id": "s1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "s1", "output": "ls output"},
            {"type": "function_call", "name": "web_search", "id": "w1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "w1", "output": "web results"},
            {"type": "message", "role": "user", "content": "continue"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None, dropped)
        # No message should reference w1
        for m in msgs:
            if m.tool_results:
                for tr in m.tool_results:
                    assert tr["tool_use_id"] != "w1"
            if m.tool_calls:
                for tc in m.tool_calls:
                    assert tc["function"]["name"] != "web_search"

    def test_only_dropped_tool_call_in_history_no_orphan_result(self):
        dropped = {"web_search"}
        inp = [
            {"type": "function_call", "name": "web_search", "id": "w1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "w1", "output": "results"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None, dropped)
        assert msgs == []

    def test_no_dropped_tool_names_arg_defaults(self):
        inp = [
            {"type": "function_call", "name": "my_tool", "id": "t1", "arguments": "{}"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        assert len(msgs) == 1
        assert msgs[0].tool_calls[0]["function"]["name"] == "my_tool"


# ==================================================================================================
# Developer messages
# ==================================================================================================

class TestDeveloperMessages:

    def test_developer_message_treated_as_non_system(self):
        inp = [
            {"type": "message", "role": "developer", "content": "You are a coder."},
            {"type": "message", "role": "user", "content": "Write code."},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        assert len(msgs) == 2
        assert any(m.role == "developer" for m in msgs)  # normalized downstream

    def test_developer_then_function_call_then_user(self):
        inp = [
            {"type": "message", "role": "developer", "content": "context"},
            {"type": "function_call", "name": "tool_x", "id": "tx1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "tx1", "output": "result"},
            {"type": "message", "role": "user", "content": "done"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        roles = [m.role for m in msgs]
        assert roles.count("assistant") == 1
        assert "user" in roles


# ==================================================================================================
# Unsupported hosted tool items in input
# ==================================================================================================

class TestSkippedItemTypes:

    def test_web_search_call_item_skipped(self):
        inp = [
            {"type": "web_search_call", "id": "x"},
            {"type": "message", "role": "user", "content": "hi"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        assert len(msgs) == 1
        assert msgs[0].content == "hi"

    def test_reasoning_item_skipped(self):
        inp = [
            {"type": "reasoning", "content": "thinking..."},
            {"type": "message", "role": "user", "content": "hello"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        assert len(msgs) == 1

    def test_computer_call_item_skipped(self):
        inp = [
            {"type": "computer_call", "id": "cc1"},
            {"type": "message", "role": "user", "content": "go"},
        ]
        sys, msgs = convert_responses_input_to_unified(inp, None)
        assert len(msgs) == 1


# ==================================================================================================
# Full round-trip: tools + input -> tool names are consistent
# ==================================================================================================

class TestFullRoundTripConsistency:
    """
    After convert_responses_tools_to_unified and convert_responses_input_to_unified,
    every tool name referenced in function_call items must exist in the unified tool list.
    """

    def _tool_names(self, unified_tools):
        if not unified_tools:
            return set()
        return {t.name for t in unified_tools}

    def _referenced_names(self, msgs):
        names = set()
        for m in msgs:
            for tc in (m.tool_calls or []):
                names.add(tc["function"]["name"])
        return names

    def test_function_only_tools(self):
        tools_raw = [{"type": "function", "name": "get_weather", "description": "d"}]
        inp = [
            {"type": "function_call", "name": "get_weather", "id": "c1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "Sunny"},
        ]
        unified_tools, dropped = convert_responses_tools_to_unified(tools_raw)
        _, msgs = convert_responses_input_to_unified(inp, None, dropped)
        assert self._referenced_names(msgs) <= self._tool_names(unified_tools)

    def test_namespace_and_function_tools(self):
        tools_raw = [
            {"type": "function", "name": "read_file", "description": "d"},
            {"type": "namespace", "name": "shell"},
        ]
        inp = [
            {"type": "function_call", "name": "shell", "id": "s1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "s1", "output": "ok"},
            {"type": "function_call", "name": "read_file", "id": "r1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "r1", "output": "contents"},
        ]
        unified_tools, dropped = convert_responses_tools_to_unified(tools_raw)
        _, msgs = convert_responses_input_to_unified(inp, None, dropped)
        assert self._referenced_names(msgs) <= self._tool_names(unified_tools)

    def test_web_search_removed_from_both_sides(self):
        tools_raw = [
            {"type": "function", "name": "get_weather"},
            {"type": "web_search", "name": "web_search"},
        ]
        inp = [
            {"type": "function_call", "name": "web_search", "id": "w1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "w1", "output": "results"},
            {"type": "function_call", "name": "get_weather", "id": "c1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "Sunny"},
        ]
        unified_tools, dropped = convert_responses_tools_to_unified(tools_raw)
        _, msgs = convert_responses_input_to_unified(inp, None, dropped)
        assert "web_search" not in self._referenced_names(msgs)
        assert "get_weather" in self._referenced_names(msgs)
        assert self._referenced_names(msgs) <= self._tool_names(unified_tools)

    def test_all_tools_dropped_history_empty(self):
        tools_raw = [{"type": "web_search", "name": "web_search"}]
        inp = [
            {"type": "function_call", "name": "web_search", "id": "w1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "w1", "output": "r"},
        ]
        unified_tools, dropped = convert_responses_tools_to_unified(tools_raw)
        _, msgs = convert_responses_input_to_unified(inp, None, dropped)
        assert msgs == []
        assert unified_tools is None
