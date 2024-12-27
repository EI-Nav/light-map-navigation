import os
import requests
import base64
import cv2
import numpy as np
from mimetypes import guess_type

# 同义千问的API配置
VLM_API_KEY: str = "sk-729ef159c2b74926874860f6e7e12ca6"
VLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLM_MODEL: str = "qwen-vl-max"

# GPT4 API配置
# VLM_API_KEY: str = "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"
# VLM_BASE_URL: str = "https://openrouter.ai/api/v1"
# VLM_MODEL: str = "gpt-4"

# 将本地图片转换为base64编码格式
def local_image_to_data_url(image):
    if isinstance(image, str):
        mime_type, _ = guess_type(image)
        with open(image, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    elif isinstance(image, np.ndarray):
        base64_encoded_data = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_encoded_data}"

# 调用Qwen-VL接口的多模态（文字+图片）请求
def gptv_response(text_prompt, image_prompt, system_prompt=""):
    headers = {
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_prompt},
        ],
        "image": local_image_to_data_url(image_prompt),
        "max_tokens": 1000
    }
    response = requests.post(f"{VLM_BASE_URL}/chat/completions", json=payload, headers=headers)
    response.raise_for_status()  # 抛出HTTP错误（如果有）
    return response.json()["choices"][0]["message"]["content"]

# 调用Qwen-VL接口的纯文本请求
def gpt_response(text_prompt, system_prompt=""):
    headers = {
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_prompt},
        ],
        "max_tokens": 1000
    }
    response = requests.post(f"{VLM_BASE_URL}/chat/completions", json=payload, headers=headers)
    response.raise_for_status()  # 抛出HTTP错误（如果有）
    return response.json()["choices"][0]["message"]["content"]
