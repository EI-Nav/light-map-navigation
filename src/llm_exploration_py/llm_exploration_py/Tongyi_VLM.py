import os
import re
import json
from dashscope import MultiModalConversation

def get_response(messages):
    """Call the qwen-vl-plus model with the provided messages and return the response."""
    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
    return response

def read_prompt_file(file_path):
    """Read the initial prompt from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    print("read successful")
    return content

def check_unit(image_path, target_unit_number, prompt_file=None):
    """Check if the unit number in the image matches the target unit number.
    
    Args:
        image_path (str): Path to the image file.
        target_unit_number (str): The unit number to check for.
        prompt_file (str): Path to the prompt file (default: None).

    Returns:
        bool: True if the unit number matches the target unit number, False otherwise.
    """
    # Set default prompt file path if not provided
    if prompt_file is None:
        prompt_file = os.path.join(os.path.dirname(__file__), 'prompt_en2.txt')

    # Read the prompt from a file
    initial_prompt = read_prompt_file(prompt_file)

    # Initialize messages with the system prompt
    messages = [{'role': 'user', 'content': initial_prompt}]
    assistant_output = get_response(messages).output.choices[0]['message']['content']
    messages.append({'role': 'assistant', 'content': assistant_output})

    # Add the image to the messages
    messages.append({'role': 'user', 'content': [{'image': image_path}]})
    assistant_output = get_response(messages).output.choices[0]['message']['content']
    messages.append({'role': 'assistant', 'content': assistant_output})

    print(assistant_output[0]['text'])

    # Extract the unit_number from the assistant_output
    try:
        # Parse the JSON string within the text field
        response_content = assistant_output[0]['text']

        json_match = re.search(r'\{.*?\}', response_content, re.DOTALL)
        # 提取并解析 JSON
        if json_match:
            json_str = json_match.group(0)
            response_content = json.loads(json_str)
            print(response_content)
        else:
            print("No JSON found in the text.")

        # Access the unit_number
        unit_number = response_content.get('unit_number')
        print(f"Extracted unit_number: {unit_number}")

        # Check if the extracted unit_number matches the target unit_number
        return unit_number == target_unit_number
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing the response: {e}")
        return False
