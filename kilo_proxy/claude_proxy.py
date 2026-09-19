"""Claude-compatible API proxy module for kilo-proxy."""

import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from kilo_proxy.config import load_config
from kilo_proxy.ip_shuffler import get_shuffler

logger = logging.getLogger("kilo-proxy")

BASE_URL = "https://api.kilo.ai/api/openrouter"


def claude_to_openai_messages(
    system: Optional[str],
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    openai_messages = []

    if system:
        openai_messages.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        text_parts.append(
                            f"[Tool call: {block.get('name', '')}({json.dumps(block.get('input', {}))})]"
                        )
                    elif btype == "tool_result":
                        text_parts.append(
                            f"[Tool result: {block.get('content', '')}]"
                        )
                    else:
                        text_parts.append(f"[{btype} content]")
                else:
                    text_parts.append(str(block))
            openai_messages.append({"role": role, "content": "\n".join(text_parts)})
        else:
            openai_messages.append({"role": role, "content": str(content)})

    return openai_messages


def openai_to_claude_response(
    openai_response: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    content = []
    finish_reason = "end_turn"

    choices = openai_response.get("choices", [])
    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        text = message.get("content", "")
        if text:
            content.append({"type": "text", "text": text})

        stop = choice.get("finish_reason", "stop")
        if stop == "tool_calls":
            finish_reason = "tool_use"
        elif stop == "length":
            finish_reason = "max_tokens"

    usage = openai_response.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content if content else [{"type": "text", "text": ""}],
        "model": model,
        "stop_reason": finish_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _sse_event(data: Dict[str, Any]) -> bytes:
    event_type = data.get("type", "message")
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def _stream_claude_completion(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    model: str,
) -> AsyncIterator[bytes]:
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}

    shuffler = get_shuffler()
    proxy = shuffler.get_current_proxy()

    client_kwargs: Dict[str, Any] = {"timeout": httpx.Timeout(300.0, connect=30.0)}
    if proxy:
        client_kwargs["proxy"] = proxy

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield _sse_event(
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
    )

    yield _sse_event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    )

    content_index = 0
    output_tokens = 0
    finish_reason = None

    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(f"Provider error {response.status_code}: {error_body.decode()[:500]}")
                    yield _sse_event({"type": "error", "error": {"type": "api_error", "message": error_body.decode()}})
                    return

                async for chunk_bytes in response.aiter_bytes(chunk_size=32):
                    if not chunk_bytes:
                        continue
                    try:
                        chunk_str = chunk_bytes.decode("utf-8")
                    except Exception:
                        continue

                    for line in chunk_str.split("\n"):
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            text = delta.get("content", "")
                            if text:
                                output_tokens += 1
                                yield _sse_event(
                                    {
                                        "type": "content_block_delta",
                                        "index": content_index,
                                        "delta": {"type": "text_delta", "text": text},
                                    }
                                )

                            fr = choices[0].get("finish_reason")
                            if fr:
                                finish_reason = fr

                        usage = chunk.get("usage")
                        if usage:
                            output_tokens = usage.get("completion_tokens", output_tokens)

    except httpx.ProxyError as e:
        yield _sse_event({"type": "error", "error": {"type": "api_error", "message": f"Proxy error: {str(e)}"}})
        return
    except Exception as e:
        yield _sse_event({"type": "error", "error": {"type": "api_error", "message": str(e)}})
        return

    yield _sse_event({"type": "content_block_stop", "index": content_index})

    stop_reason = "end_turn"
    if finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"

    yield _sse_event(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
    )

    yield _sse_event({"type": "message_stop"})


async def create_claude_completion(
    system: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 4096,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: bool = False,
) -> Union[Dict[str, Any], StreamingResponse]:
    if messages is None:
        messages = []

    openai_messages = claude_to_openai_messages(system, messages)

    body: Dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": max_tokens,
    }

    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if stop_sequences:
        body["stop"] = stop_sequences

    config = load_config()
    auth_token = config.auth_token or "anonymous"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
    }

    url = f"{BASE_URL}/chat/completions"

    if stream:
        return StreamingResponse(
            _stream_claude_completion(url, headers, body, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    shuffler = get_shuffler()
    proxy = shuffler.get_current_proxy()

    client_kwargs: Dict[str, Any] = {"timeout": httpx.Timeout(300.0, connect=30.0)}
    if proxy:
        client_kwargs["proxy"] = proxy

    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.post(url, headers=headers, json=body)
        if response.status_code != 200:
            logger.error(f"Provider error {response.status_code}: {response.text[:500]}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Claude completion failed: {response.text}",
            )
        openai_response = response.json()
        return openai_to_claude_response(openai_response, model)
