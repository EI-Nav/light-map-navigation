import os
import openai
import base64
import cv2
import numpy as np
from mimetypes import guess_type

# 设置 API 密钥和 Base URL
api_key = "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"
base_url = "https://openrouter.ai/api/v1"
MODEL_NAME = "gpt-4"

# 初始化 OpenAI 客户端
openai.api_key = api_key
openai.api_base = base_url  # 如果没有 base_url，通常可以不设置，直接使用默认值

# 将本地图片转换为 Base64 编码格式
def local_image_to_data_url(image):
    if isinstance(image, str):
        mime_type, _ = guess_type(image)
        with open(image, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    elif isinstance(image, np.ndarray):
        base64_encoded_data = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_encoded_data}"

# 调用 OpenAI GPT-4 Vision（多模态）接口
def gptv_response(text_prompt, image_prompt, system_prompt=""):
    prompt = [{'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': [{'type': 'text', 'text': text_prompt},
                                          {'type': 'image_url', 'image_url': {'url': local_image_to_data_url(image_prompt)}}]}]
    
    response = openai.ChatCompletion.create(
        model= MODEL_NAME,  # 使用 GPT-4 模型
        messages=prompt,
        max_tokens=1000
    )
    
    return response.choices[0].message['content']

# 调用 OpenAI GPT-4（纯文本）接口
def gpt_response(text_prompt, system_prompt=""):
    prompt = [{'role': 'system', 'content': system_prompt},
              {'role': 'user', 'content': [{'type': 'text', 'text': text_prompt}]}]
    
    response = openai.ChatCompletion.create(
        model= MODEL_NAME,  # 使用 GPT-4 模型
        messages=prompt,
        max_tokens=1000
    )
    
    return response.choices[0].message['content']
