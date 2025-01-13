import os
from openai import AzureOpenAI,OpenAI
import requests
import base64
import cv2
import numpy as np
from mimetypes import guess_type


gpt4_client = OpenAI(
    api_key='sk-5dba84af311d4a0f8045a969086bfa46', 
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)
GPT4_MODEL = "qwen-vl-max"

gpt4v_client = OpenAI(
    api_key='sk-5dba84af311d4a0f8045a969086bfa46', 
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)
GPT4V_MODEL = "qwen-vl-max"

def local_image_to_data_url(image):
    if isinstance(image,str):
        mime_type, _ = guess_type(image)
        with open(image, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    elif isinstance(image,np.ndarray):
        base64_encoded_data = base64.b64encode(cv2.imencode('.jpg',image)[1]).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_encoded_data}"

def gptv_response(text_prompt,image_prompt,system_prompt=""):
    prompt = [{'role':'system','content':system_prompt},
             {'role':'user','content':[{'type':'text','text':text_prompt},
                                       {'type':'image_url','image_url':{'url':local_image_to_data_url(image_prompt)}}]}]
    response = gpt4v_client.chat.completions.create(model=GPT4V_MODEL,
                                                    messages=prompt,
                                                    max_tokens=1000)
    return response.choices[0].message.content

def gpt_response(text_prompt,system_prompt=""):
    prompt = [{'role':'system','content':system_prompt},
              {'role':'user','content':[{'type':'text','text':text_prompt}]}]
    response = gpt4_client.chat.completions.create(model=GPT4_MODEL,
                                              messages=prompt,
                                              max_tokens=1000)
    return response.choices[0].message.content