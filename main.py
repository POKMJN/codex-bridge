# main.py
# OpenAI Responses API -> Anthropic Messages API 转换代理
# 支持 SSE 流式响应 + 工具调用

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import os
import sys
import time
import json
import asyncio

from dotenv import load_dotenv
load_dotenv()

import anthropic

# 启动时校验必需的环境变量
API_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "")

if not API_KEY:
    print("[FATAL] ANTHROPIC_AUTH_TOKEN not set in .env or environment. Exiting.")
    sys.exit(1)
if not DEFAULT_MODEL:
    print("[FATAL] ANTHROPIC_MODEL not set in .env or environment. Exiting.")
    sys.exit(1)

print(f"[PROXY] Loaded model: {DEFAULT_MODEL}")

app = FastAPI()

# Optional: load WeChat webhook router if available
try:
    from routes.wechat import router as wechat_router
    app.include_router(wechat_router)
except Exception as e:
    print(f"[INFO] WeChat router not loaded: {e}")

ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")


def convert_input_to_anthropic(body):
    """把 OpenAI Responses API 输入转成 Anthropic messages

    处理:
      - 普通 user/assistant 文本消息
      - developer (system) 消息 -> 提取为 system prompt
      - function_call 项 -> assistant tool_use content block
      - function_call_output 项 -> user tool_result content block
      - Anthropic 要求严格的 user/assistant 交替，同角色相邻的会被合并
    """
    developer_content = ""
    input_val = body.get("input", "")

    if isinstance(input_val, str):
        return [{"role": "user", "content": input_val}], ""

    if not isinstance(input_val, list):
        return [{"role": "user", "content": str(input_val)}], ""

    # 第一步: 把每个 item 转成 (role, content) 对
    items = []
    for item in input_val:
        if not isinstance(item, dict):
            items.append(("user", str(item)))
            continue

        item_type = item.get("type", "")
        role = item.get("role", "")

        # developer/system 消息
        if role == "developer":
            content = item.get("content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "input_text"]
                content = "\n".join(texts)
            elif not isinstance(content, str):
                content = str(content)
            developer_content = content
            continue

        # function_call -> assistant tool_use
        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id", "")
            name = item.get("name", "")
            arguments = item.get("arguments", "{}")
            try:
                input_data = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (json.JSONDecodeError, TypeError):
                input_data = {"raw": str(arguments)}
            items.append(("assistant", {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": input_data,
            }))
            continue

        # function_call_output -> user tool_result
        if item_type == "function_call_output":
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            items.append(("user", {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": str(output),
            }))
            continue

        # 普通 message (role based)
        if role in ("user", "assistant"):
            content = item.get("content", "")
            if isinstance(content, str):
                items.append((role, content))
            elif isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict):
                        bt = block.get("type", "")
                        if bt in ("input_text", "output_text", "text"):
                            texts.append(block.get("text", ""))
                if texts:
                    items.append((role, "\n".join(texts)))
            else:
                items.append((role, str(content)))
            continue

        # input_text 无 role
        if item_type == "input_text":
            items.append(("user", item.get("text", "")))
            continue

        # fallback
        items.append(("user", str(item)))

    # 第二步: 合并连续同角色的项 (Anthropic 要求严格交替)
    anthropic_messages = []
    for role, content in items:
        if anthropic_messages and anthropic_messages[-1]["role"] == role:
            existing = anthropic_messages[-1]["content"]
            # 需要合并 -> 统一转成 list 格式
            if isinstance(existing, str):
                if isinstance(content, str):
                    anthropic_messages[-1]["content"] = existing + "\n" + content
                else:
                    anthropic_messages[-1]["content"] = [
                        {"type": "text", "text": existing},
                        content,
                    ]
            elif isinstance(existing, list):
                if isinstance(content, str):
                    existing.append({"type": "text", "text": content})
                else:
                    existing.append(content)
        else:
            # dict 类型的 content (tool_use/tool_result) 必须包在 list 里
            if isinstance(content, dict):
                anthropic_messages.append({"role": role, "content": [content]})
            else:
                anthropic_messages.append({"role": role, "content": content})

    # 确保第一条是 user (Anthropic 要求)
    if anthropic_messages and anthropic_messages[0]["role"] != "user":
        anthropic_messages.insert(0, {"role": "user", "content": "(continue)"})

    return anthropic_messages, developer_content


def convert_tools_to_anthropic(tools):
    """把 OpenAI tools 格式转成 Anthropic tools 格式"""
    if not tools:
        return None

    anthropic_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type")

        # Only convert function-type tools; skip web_search, code_interpreter, etc.
        if tool_type != "function":
            print(f"[PROXY] Skipping non-function tool: type={tool_type}")
            continue

        anthropic_tool = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {"type": "object", "properties": {}})
        }

        if "function" in tool:
            anthropic_tool["name"] = tool["function"].get("name", anthropic_tool["name"])
            anthropic_tool["description"] = tool["function"].get("description", anthropic_tool["description"])
            anthropic_tool["input_schema"] = tool["function"].get("parameters", anthropic_tool["input_schema"])

        if not anthropic_tool["name"]:
            print(f"[PROXY] Skipping tool without name: {tool}")
            continue

        if not anthropic_tool["input_schema"]:
            anthropic_tool["input_schema"] = {"type": "object", "properties": {}}
        elif "type" not in anthropic_tool["input_schema"]:
            anthropic_tool["input_schema"]["type"] = "object"

        anthropic_tools.append(anthropic_tool)

    return anthropic_tools if anthropic_tools else None


def anthropic_to_openai_stream(event, msg_id=None, model=None, state=None):
    """把 Anthropic 流式事件转成 OpenAI Responses SSE 格式

    事件序列 (文本):
      response.created -> output_item.added -> content_part.added
      -> output_text.delta (多次) -> output_text.done
      -> content_part.done -> output_item.done -> response.completed

    事件序列 (工具调用):
      response.created -> output_item.added (function_call)
      -> function_call_arguments.delta (多次) -> function_call_arguments.done
      -> output_item.done -> response.completed
    """
    if state is None:
        state = {
            "text_block_indices": set(),
            "tool_use_indices": {},   # idx -> {id, name, arguments}
            "done_indices": set(),
            "accumulated_text": {},   # idx -> text
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "output_index_counter": 0,  # OpenAI output 索引计数器
            "idx_to_output_index": {},  # Anthropic idx -> OpenAI output_index
        }
    event_type = event.type

    if event_type == "message_start":
        msg_id = event.message.id
        model = event.message.model
        if hasattr(event.message, 'usage') and event.message.usage:
            state["usage"]["input_tokens"] = getattr(event.message.usage, 'input_tokens', 0)
        data = {
            "type": "response.created",
            "response": {
                "id": msg_id,
                "object": "response",
                "model": model,
                "created_at": int(time.time()),
                "status": "in_progress",
                "output": [],
            }
        }
        return msg_id, model, f"data: {json.dumps(data)}\n\n", state

    elif event_type == "content_block_start":
        block = event.content_block
        idx = event.index

        if block.type == "text":
            # 分配 output_index
            out_idx = state["output_index_counter"]
            state["output_index_counter"] += 1
            state["idx_to_output_index"][idx] = out_idx

            state["text_block_indices"].add(idx)
            state["accumulated_text"][idx] = ""
            item_id = f"msg_{msg_id}_{out_idx}"
            sse = ""
            sse += f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': out_idx, 'item': {'id': item_id, 'type': 'message', 'role': 'assistant', 'status': 'in_progress', 'content': []}})}\n\n"
            sse += f"data: {json.dumps({'type': 'response.content_part.added', 'item_id': item_id, 'output_index': out_idx, 'content_index': 0, 'part': {'type': 'output_text', 'text': ''}})}\n\n"
            return msg_id, model, sse, state

    elif event_type == "content_block_delta":
        delta = event.delta
        idx = event.index

        if delta.type == "text_delta":
            text = getattr(delta, "text", "") or ""
            if idx in state["accumulated_text"]:
                state["accumulated_text"][idx] += text
            out_idx = state["idx_to_output_index"].get(idx, idx)
            data = {
                "type": "response.output_text.delta",
                "item_id": f"msg_{msg_id}_{out_idx}",
                "output_index": out_idx,
                "content_index": 0,
                "delta": text,
            }
            return msg_id, model, f"data: {json.dumps(data)}\n\n", state

    elif event_type == "message_stop":
        # 构建完整的 response.completed 事件
        output = []

        # 按 output_index 排序收集所有输出项
        all_items = []

        for idx in state["text_block_indices"]:
            out_idx = state["idx_to_output_index"].get(idx, idx)
            text = state["accumulated_text"].get(idx, "")
            all_items.append((out_idx, {
                "type": "message",
                "id": f"msg_{msg_id}_{out_idx}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}]
            }))

        for idx, tool_info in state["tool_use_indices"].items():
            out_idx = state["idx_to_output_index"].get(idx, idx)
            all_items.append((out_idx, {
                "type": "function_call",
                "id": f"fc_{msg_id}_{out_idx}",
                "call_id": tool_info["id"],
                "name": tool_info["name"],
                "arguments": tool_info["arguments"],
                "status": "completed",
            }))

        all_items.sort(key=lambda x: x[0])
        output = [item for _, item in all_items]

        usage = state["usage"]
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        response_obj = {
            "id": msg_id,
            "object": "response",
            "model": model,
            "created_at": int(time.time()),
            "status": "completed",
            "output": output,
            "usage": usage,
        }
        data = {
            "type": "response.completed",
            "response": response_obj,
        }
        return msg_id, model, f"data: {json.dumps(data)}\n\n", state

    return msg_id, model, "", state


async def stream_response(request_id, model, anthropic_messages, max_tokens, temperature, instructions, api_key, tools=None):
    """流式转发 Anthropic 响应到客户端"""
    anthropic_client = anthropic.Anthropic(api_key=api_key, base_url=ANTHROPIC_BASE_URL)

    anthropic_params = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        anthropic_params["temperature"] = temperature
    if instructions:
        anthropic_params["system"] = instructions
    if tools:
        anthropic_params["tools"] = tools
        print(f"[PROXY] Using {len(tools)} tools")

    print(f"[PROXY] streaming to Anthropic: model={model}, msgs={len(anthropic_messages)}")
    print(f"[PROXY] Anthropic messages: {json.dumps(anthropic_messages, ensure_ascii=False)[:500]}")

    msg_id = None
    model_name = None
    state = None  # will be initialized by anthropic_to_openai_stream
    try:
        with anthropic_client.messages.stream(**anthropic_params) as stream:
            for event in stream:
                print(f"[PROXY] RAW EVENT: {event.type}")
                try:
                    msg_id, model_name, sse_data, state = anthropic_to_openai_stream(event, msg_id, model_name, state)
                    if sse_data:
                        print(f"[PROXY] SSE OUT: {sse_data[:150]}")
                        yield sse_data
                except Exception as e:
                    print(f"[PROXY] event error: {e}")
                    import traceback
                    traceback.print_exc()
    except Exception as e:
        print(f"[PROXY] stream error: {e}")
        import traceback
        traceback.print_exc()
        error_response = {
            "id": msg_id or f"err_{int(time.time())}",
            "object": "response",
            "model": model_name or model,
            "created_at": int(time.time()),
            "status": "failed",
            "error": {"message": str(e), "type": "server_error"},
            "output": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        yield f'data: {json.dumps({"type": "response.completed", "response": error_response})}\n\n'


@app.post("/v1/responses")
@app.post("/responses")
async def handle_responses(request: Request):
    """接收 OpenAI Responses API 请求，支持流式和普通模式"""
    body = await request.json()
    print(f"[PROXY] incoming: {json.dumps(body, ensure_ascii=False)[:500]}")

    model = body.get("model", DEFAULT_MODEL)
    max_output_tokens = body.get("max_output_tokens") or body.get("max_tokens") or 4096
    temperature = body.get("temperature")
    instructions = body.get("instructions") or body.get("system")
    stream = body.get("stream", False)
    tools = body.get("tools")

    anthropic_messages, developer_content = convert_input_to_anthropic(body)
    anthropic_tools = convert_tools_to_anthropic(tools) if tools else None

    if developer_content:
        if instructions:
            instructions = developer_content + "\n" + instructions
        else:
            instructions = developer_content

    api_key = API_KEY

    # 流式模式
    if stream:
        print(f"[PROXY] streaming mode")
        return StreamingResponse(
            stream_response(
                request_id=body.get("id", "unknown"),
                model=model,
                anthropic_messages=anthropic_messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
                instructions=instructions,
                api_key=api_key,
                tools=anthropic_tools,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": body.get("id", ""),
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked",
            }
        )

    # 非流式模式
    print(f"[PROXY] non-stream mode: model={model}, msgs={len(anthropic_messages)}")

    anthropic_client = anthropic.Anthropic(api_key=API_KEY, base_url=ANTHROPIC_BASE_URL)
    anthropic_params = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_output_tokens,
    }
    if temperature is not None:
        anthropic_params["temperature"] = temperature
    if instructions:
        anthropic_params["system"] = instructions
    if anthropic_tools:
        anthropic_params["tools"] = anthropic_tools

    try:
        resp = anthropic_client.messages.create(**anthropic_params)
    except anthropic.BadRequestError as e:
        print(f"[PROXY] Anthropic error: {e}")
        return {"error": {"message": str(e), "type": "invalid_request_error"}}, 400
    except Exception as e:
        print(f"[PROXY] Error: {e}")
        return {"error": {"message": str(e), "type": "internal_error"}}, 500

    print(f"[PROXY] <- Anthropic: id={resp.id}, stop_reason={resp.stop_reason}")

    # 构建 output
    output = []
    out_idx = 0
    for block in (resp.content or []):
        if hasattr(block, "type"):
            if block.type == "text":
                output.append({
                    "type": "message",
                    "id": f"msg_{resp.id}_{out_idx}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": block.text}]
                })
                out_idx += 1
            elif block.type == "tool_use":
                output.append({
                    "type": "function_call",
                    "id": f"fc_{resp.id}_{out_idx}",
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": json.dumps(block.input),
                    "status": "completed",
                })
                out_idx += 1

    usage = {
        "input_tokens": getattr(resp.usage, "input_tokens", 0),
        "output_tokens": getattr(resp.usage, "output_tokens", 0),
        "total_tokens": getattr(resp.usage, "input_tokens", 0) + getattr(resp.usage, "output_tokens", 0),
    }

    return {
        "id": resp.id,
        "object": "response",
        "model": resp.model,
        "created_at": int(time.time()),
        "status": "completed",
        "output": output,
        "usage": usage,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
