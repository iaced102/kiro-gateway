# -*- coding: utf-8 -*-

"""
FastAPI routes for OpenAI Responses API.

Implements POST /v1/responses compatible with OpenAI's Responses API.
"""

import json
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from kiro.config import PROXY_API_KEY, PROFILE_ARN
from kiro.models_responses import CreateResponseRequest
from kiro.converters_responses import build_kiro_payload_from_responses
from kiro.streaming_responses import (
    stream_with_first_token_retry_responses,
    collect_responses_response,
)
from kiro.http_client import KiroHttpClient
from kiro.utils import generate_conversation_id

try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# --- Security ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(auth_header: str = Security(api_key_header)) -> bool:
    if not auth_header or auth_header != f"Bearer {PROXY_API_KEY}":
        logger.warning("Access attempt with invalid API key (Responses endpoint)")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


router = APIRouter(tags=["Responses API"])


@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def create_response(
    request: Request,
    request_data: CreateResponseRequest,
):
    """
    OpenAI Responses API endpoint.

    Accepts requests in Responses API format and translates them to Kiro API.
    Supports both streaming and non-streaming modes.
    """
    logger.info(f"Request to /v1/responses (model={request_data.model}, stream={request_data.stream})")

    # ==============================================================================
    # Account System: Failover or Legacy
    # ==============================================================================

    if request.app.state.account_system:
        from kiro.account_errors import classify_error, ErrorType

        account_manager = request.app.state.account_manager
        all_accounts = list(account_manager._accounts.keys())
        MAX_ATTEMPTS = len(all_accounts) * 2

        last_error_message = None
        last_error_status = None
        tried_accounts = set()

        for attempt in range(MAX_ATTEMPTS):
            account = await account_manager.get_next_account(
                request_data.model,
                exclude_accounts=tried_accounts
            )

            if account is None:
                if len(all_accounts) == 1:
                    return JSONResponse(
                        status_code=last_error_status or 503,
                        content={"error": {"message": last_error_message or "Account unavailable", "type": "api_error"}},
                    )
                detail = "No available accounts for this model."
                if last_error_message:
                    detail += f" Last error: {last_error_message}"
                return JSONResponse(status_code=503, content={"error": {"message": detail, "type": "api_error"}})

            tried_accounts.add(account.id)
            auth_manager = account.auth_manager
            model_cache = account.model_cache

            conversation_id = generate_conversation_id()
            profile_arn_for_payload = auth_manager.profile_arn or PROFILE_ARN or ""

            try:
                kiro_payload, _namespace_map = build_kiro_payload_from_responses(
                    request_data, conversation_id, profile_arn_for_payload
                )
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": {"message": str(e), "type": "invalid_request_error"}})

            url = f"{auth_manager.api_host}/generateAssistantResponse"
            logger.debug(f"Kiro API URL: {url} (account: {account.id})")

            if request_data.stream:
                http_client = KiroHttpClient(auth_manager, shared_client=None)
            else:
                http_client = KiroHttpClient(auth_manager, shared_client=request.app.state.http_client)

            # Prepare for token counting
            input_items = request_data.input if isinstance(request_data.input, list) else [{"role": "user", "content": request_data.input}]
            tools_for_tokenizer = request_data.tools

            try:
                response = await http_client.request_with_retry("POST", url, kiro_payload, stream=True)

                if response.status_code == 200:
                    await account_manager.report_success(account.id, request_data.model)

                    if request_data.stream:
                        async def stream_wrapper():
                            streaming_error = None
                            try:
                                async def make_retry_request():
                                    return await http_client.request_with_retry("POST", url, kiro_payload, stream=True)

                                async for chunk in stream_with_first_token_retry_responses(
                                    make_request=make_retry_request,
                                    model=request_data.model,
                                    model_cache=model_cache,
                                    auth_manager=auth_manager,
                                    initial_response=response,
                                    request_messages=input_items,
                                    request_tools=tools_for_tokenizer,
                                    namespace_map=_namespace_map,
                                ):
                                    yield chunk
                            except GeneratorExit:
                                logger.debug("Client disconnected (Responses streaming)")
                            except Exception as e:
                                streaming_error = e
                                try:
                                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
                                except Exception:
                                    pass
                            finally:
                                await http_client.close()
                                if streaming_error:
                                    logger.error(f"HTTP 500 - POST /v1/responses (streaming) - {str(streaming_error)[:100]}")
                                else:
                                    logger.info("HTTP 200 - POST /v1/responses (streaming) - completed")

                        return StreamingResponse(
                            stream_wrapper(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                        )

                    else:
                        resp_data = await collect_responses_response(
                            response, request_data.model, model_cache, auth_manager,
                            request_messages=input_items, request_tools=tools_for_tokenizer,
                            namespace_map=_namespace_map,
                        )
                        await http_client.close()
                        logger.info("HTTP 200 - POST /v1/responses (non-streaming) - completed")
                        return JSONResponse(content=resp_data)

                else:
                    try:
                        error_content = await response.aread()
                    except Exception:
                        error_content = b"Unknown error"
                    await http_client.close()
                    error_text = error_content.decode("utf-8", errors="replace")

                    error_reason = None
                    try:
                        error_json = json.loads(error_text)
                        from kiro.kiro_errors import enhance_kiro_error
                        error_info = enhance_kiro_error(error_json)
                        error_reason = error_info.reason
                        last_error_message = error_info.user_message
                        last_error_status = response.status_code
                    except (json.JSONDecodeError, KeyError):
                        last_error_message = error_text
                        last_error_status = response.status_code

                    error_type = classify_error(response.status_code, error_reason)
                    await account_manager.report_failure(account.id, request_data.model, error_type, response.status_code, error_reason)

                    if error_type == ErrorType.FATAL or len(all_accounts) == 1:
                        return JSONResponse(
                            status_code=response.status_code,
                            content={"error": {"message": last_error_message, "type": "api_error"}},
                        )
                    continue

            except HTTPException as e:
                await http_client.close()
                if e.status_code in (502, 504):
                    await account_manager.report_failure(account.id, request_data.model, ErrorType.RECOVERABLE, e.status_code, None)
                    last_error_message = str(e.detail)
                    last_error_status = e.status_code
                    if len(all_accounts) == 1:
                        break
                    continue
                raise
            except Exception as e:
                await http_client.close()
                logger.error(f"Internal error: {e}", exc_info=True)
                return JSONResponse(status_code=500, content={"error": {"message": f"Internal Server Error: {str(e)}", "type": "api_error"}})

        # All attempts exhausted
        detail = last_error_message or "All accounts failed"
        return JSONResponse(
            status_code=last_error_status or 503,
            content={"error": {"message": detail, "type": "api_error"}},
        )

    else:
        # ==============================================================================
        # Legacy mode
        # ==============================================================================
        account = request.app.state.account_manager.get_first_account()
        if not account.auth_manager:
            return JSONResponse(status_code=503, content={"error": {"message": "No initialized accounts available", "type": "api_error"}})

        auth_manager = account.auth_manager
        model_cache = account.model_cache

        conversation_id = generate_conversation_id()
        profile_arn_for_payload = auth_manager.profile_arn or PROFILE_ARN or ""

        try:
            kiro_payload, _namespace_map = build_kiro_payload_from_responses(
                request_data, conversation_id, profile_arn_for_payload
            )
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": {"message": str(e), "type": "invalid_request_error"}})

        url = f"{auth_manager.api_host}/generateAssistantResponse"

        if request_data.stream:
            http_client = KiroHttpClient(auth_manager, shared_client=None)
        else:
            http_client = KiroHttpClient(auth_manager, shared_client=request.app.state.http_client)

        input_items = request_data.input if isinstance(request_data.input, list) else [{"role": "user", "content": request_data.input}]
        tools_for_tokenizer = request_data.tools

        try:
            response = await http_client.request_with_retry("POST", url, kiro_payload, stream=True)

            if response.status_code != 200:
                try:
                    error_content = await response.aread()
                except Exception:
                    error_content = b"Unknown error"
                await http_client.close()
                error_text = error_content.decode("utf-8", errors="replace")
                error_message = error_text
                try:
                    from kiro.kiro_errors import enhance_kiro_error
                    error_info = enhance_kiro_error(json.loads(error_text))
                    error_message = error_info.user_message
                except Exception:
                    pass
                logger.warning(f"HTTP {response.status_code} - POST /v1/responses - {error_message[:100]}")
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": {"message": error_message, "type": "api_error"}},
                )

            if request_data.stream:
                async def stream_wrapper():
                    streaming_error = None
                    try:
                        async def make_retry_request():
                            return await http_client.request_with_retry("POST", url, kiro_payload, stream=True)

                        async for chunk in stream_with_first_token_retry_responses(
                            make_request=make_retry_request,
                            model=request_data.model,
                            model_cache=model_cache,
                            auth_manager=auth_manager,
                            initial_response=response,
                            request_messages=input_items,
                            request_tools=tools_for_tokenizer,
                            namespace_map=_namespace_map,
                        ):
                            yield chunk
                    except GeneratorExit:
                        logger.debug("Client disconnected (Responses streaming)")
                    except Exception as e:
                        streaming_error = e
                        try:
                            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
                        except Exception:
                            pass
                    finally:
                        await http_client.close()
                        if streaming_error:
                            logger.error(f"HTTP 500 - POST /v1/responses (streaming) - {str(streaming_error)[:100]}")
                        else:
                            logger.info("HTTP 200 - POST /v1/responses (streaming) - completed")

                return StreamingResponse(
                    stream_wrapper(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            else:
                resp_data = await collect_responses_response(
                    response, request_data.model, model_cache, auth_manager,
                    request_messages=input_items, request_tools=tools_for_tokenizer,
                    namespace_map=_namespace_map,
                )
                await http_client.close()
                logger.info("HTTP 200 - POST /v1/responses (non-streaming) - completed")
                return JSONResponse(content=resp_data)

        except HTTPException as e:
            await http_client.close()
            logger.error(f"HTTP {e.status_code} - POST /v1/responses - {e.detail}")
            raise
        except Exception as e:
            await http_client.close()
            logger.error(f"Internal error: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": {"message": f"Internal Server Error: {str(e)}", "type": "api_error"}})
