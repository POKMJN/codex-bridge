import os
import time
import hashlib
import xml.etree.ElementTree as ET
from typing import Optional, Dict, List
from fastapi import APIRouter, Request, Response, HTTPException
import httpx

router = APIRouter()
WECHAT_TOKEN = os.getenv("WECHAT_TOKEN", "your-token")
CODEX_BRIDGE_URL = os.getenv("CODEX_BRIDGE_URL", "http://127.0.0.1:8000")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "MiniMax-M2.7")

# Simple in-memory session store (replace with Redis for production)
conversations: Dict[str, List[str]] = {}
MAX_HISTORY = 6  # last N messages (user+assistant pairs)


def verify_wechat_signature(token: str, timestamp: str, nonce: str, signature: str) -> bool:
    s = "".join(sorted([token, timestamp, nonce]))
    sha1 = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return sha1 == signature


def parse_wechat_xml(body: bytes) -> Dict[str, str]:
    root = ET.fromstring(body)
    data = {}
    for child in root:
        data[child.tag] = child.text or ""
    return data


def build_text_reply(to_user: str, from_user: str, content: str) -> str:
    xml = f"""<xml>
  <ToUserName><![CDATA[{to_user}]]></ToUserName>
  <FromUserName><![CDATA[{from_user}]]></FromUserName>
  <CreateTime>{int(time.time())}</CreateTime>
  <MsgType><![CDATA[text]]></MsgType>
  <Content><![CDATA[{content}]]></Content>
</xml>"""
    return xml


async def call_codex_bridge(user_text: str, openid: str) -> str:
    # Include short history if available
    history = conversations.get(openid, [])
    prompt_parts = []
    for i, item in enumerate(history[-MAX_HISTORY:]):
        prefix = "User: " if i % 2 == 0 else "Assistant: "
        prompt_parts.append(prefix + item)
    prompt_parts.append("User: " + user_text)
    prompt = "\n".join(prompt_parts)

    payload = {
        "model": ANTHROPIC_MODEL,
        "input": prompt
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{CODEX_BRIDGE_URL}/responses", json=payload)
        resp.raise_for_status()
        j = resp.json()
    text = ""
    try:
        text = j.get("output", [])[0].get("content", [])[0].get("text", "")
    except Exception:
        text = j.get("output_text") or j.get("text") or ""
    if not text:
        text = str(j)
    return text


@router.get("/wechat")
async def wechat_verify(signature: str, timestamp: str, nonce: str, echostr: Optional[str] = None):
    if not verify_wechat_signature(WECHAT_TOKEN, timestamp, nonce, signature):
        raise HTTPException(status_code=403, detail="invalid signature")
    return Response(content=echostr or "", media_type="text/plain")


@router.post("/wechat")
async def wechat_message(request: Request, signature: str, timestamp: str, nonce: str):
    body = await request.body()
    if not verify_wechat_signature(WECHAT_TOKEN, timestamp, nonce, signature):
        raise HTTPException(status_code=403, detail="invalid signature")
    data = parse_wechat_xml(body)
    msg_type = data.get("MsgType", "")
    from_user = data.get("FromUserName", "")
    to_user = data.get("ToUserName", "")

    if msg_type == "text":
        user_text = data.get("Content", "").strip()
        conv = conversations.setdefault(from_user, [])
        conv.append(user_text)
        try:
            reply = await call_codex_bridge(user_text, from_user)
        except Exception:
            reply = "系统繁忙，请稍后重试。"
        conv.append(reply)
        if len(conv) > MAX_HISTORY * 2:
            conversations[from_user] = conv[-(MAX_HISTORY*2):]
        xml = build_text_reply(from_user, to_user, reply)
        return Response(content=xml, media_type="application/xml; charset=utf-8")
    else:
        return Response(content="success", media_type="text/plain")
