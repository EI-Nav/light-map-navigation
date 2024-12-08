building_coordinates = {
    "building1": (10.511010327, 0.000235218),
    "building2": (10.511016244, 0.000086922),
    "building3": (10.510963920, -0.000079640),
    "building4": (10.510953840, -0.000173880),
    "building5": (10.511022280, -0.000178800),
    "building6": (10.510985680, -0.000295100),
    "building7": (10.511262554, 0.000258062),
    "building8": (10.511312240, 0.000089240),
    "building9": (10.511525424, 0.000177076),
    "building10": (10.511256380, -0.000125360),
    "building11": (10.511509522, -0.000094233),
    "building12": (10.511246718, -0.000243509),
    "building13": (10.511806045, 0.000213527),
    "building14": (10.512026964, 0.000253336),
    "building15": (10.511943636, 0.000122945),
    "building16": (10.512097940, 0.000131020),
    "building17": (10.512033585, -0.000183692),
}

# for real-world
# building_coordinates = {
#     "building1": (116.416246613,39.971315650),
#     "building2": (116.415284417,39.972169284),
# }

units_coordinates = {
    "building1": {
        "unit1": (),
        "unit2": (),
        "unit3": (),
        "unit4": (),
        "unit5": (),
    },
    "building2": {
        "unit1": (),
        "unit2": (),
    },
    "building3": {
        "unit1": (),
        "unit2": (),
    },
}

# for real-world
# units_coordinates = {
#     "building1": {
#         "unit1": (),
#         "unit2": (),
#         "unit3": (),
#         "unit4": (),
#         "unit5": (),
#     }
# }

# user_input = "请帮我将苹果送至一号楼五单元，并且将橘子送至一号楼六单元。"    
output_format =  """
{
    "thinking_process":  {
        "user_input_english": ""
        "orin_unit": {
            "item1": "building1 unit2",  
            "item2": "building1 unit5",
            "item3": "building7 unit1"   
        },
        "chain_of_thoughts": ""}
    "deliveries": [
        {
            "item": "待配送物品",
            "building_ids": ["building1"],
            "building_coordinates": [10.511010327, 0.000235218],
            "unit_ids": ["unit2"],
            "specific_unit_coordinates_at_this_building": ["unknown"],
            "unit_coordinates": []
        },
        {
            "item": "待配送物品",
            "building_ids": ["building1"],
            "building_coordinates": [10.511010327, 0.000235218],
            "unit_ids": ["unit5"],
            "specific_unit_coordinates_at_this_building": ["unknown"],
            "unit_coordinates": [3.8933, 4.3390]
        },
        {
            "item": "待配送物品",
            "destination": "building7",
            "building_coordinates": [10.511262554, 0.000258062],
            "unit_ids": ["unit1"],
            "specific_unit_coordinates_at_this_building": ["unknown"],
            "unit_coordinates": []
        }
    ]
}

"""


output_format_explain = """
在这个JSON输出中，
- `deliveries`包含了（可能发生：为了高效运输而调整顺序后的）待配送的物品及其目的地和坐标。
    -`unit_ids`为具体单元的编号，每次只能填写一个编号,即不能是["unit2", "unit3"]的数组格式，并且每个具体单元只经过一次。
    -`unit_coordinates`为具体单元的坐标，如果不知道具体单元的坐标，则为空。
    -`specific_unit_coordinates_at_this_building`为具体单元坐标是否已知，如果已知，填写"known"，否则填写"unknown"。当unknown时，可以之后调用感知模块获取，不需要决策模块负责。
- `thinking_process`中需要包含
    -`user_input_english`将用户的输入翻译为英文
    -`orin_building_unit`从用户输入中提取的目标单元名称
    -`chain of thoughts`自己的思考过程
"""
# 定义黄色的 ANSI 转义序列
yellow = '\033[93m'
green = '\033[92m'
reset = '\033[0m'

# 2.将多模态信息传给GPT-4o Mini
from openai import OpenAI
import json

def call_llm(user_input):
    print(f"{green}{user_input}{reset}")
    client = OpenAI(
        api_key = "sk-proj-Dz7viFcxpW3DOzmYnK4qT3BlbkFJi8SC31U5WW1mHN5CzYKr"
    )
    
    messages = [
        {"role": "system", "content": "你是一个快递配送移动机器人的决策模块.有the state of art的motion planning能力。Before achieving this goal,you need to devide the final goal into subtasks and solve them step by step.输出格式可以由python json.loads()成功加载"},
        {"role": "user", "content": [
            {"type": "text", "text": f"用户输入为：{user_input}\n"},
            {"type": "text", "text": "将用户的输入翻译为英文，单元为unit,号楼为building。"},
            {"type": "text", "text": "原则一：你可以调整配送顺序，为了使配送更加高效，你可以同一地点一次性配送多个物品。"},
            {"type": "text", "text": "原则二：你不可以自己添加新的单元目标地点，只能在用户输入的单元做选择。"},
            {"type": "text", "text": f"已知building_coordinates：{building_coordinates},units_coordinates：{units_coordinates}，"},
            {"type": "text", "text": f"输出格式仿照{output_format},输出格式解释：{output_format_explain}"},
            # {"type":"text","text":f"你的起点在building1，途经点为building4和building7，请抉择下应该如何先到达building4还是building7？并给出理由。已知{coordinates}"},
            # *[{"type":"image_url","image_url":{"url":url}} for url in image_urls]
        ]}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )
    print(f"{yellow}{response.choices[0].message.content}{reset}")
    return response.choices[0].message.content

def extract_coordinates(input_json):
    """
    解析 JSON 字符串并提取 building_coordinates 和 unit_coordinates。

    参数:
    input_json (str): JSON 字符串，包含 deliveries 数据。

    返回:
    tuple: 两个列表，第一个列表包含所有的 building_coordinates，第二个列表包含所有的 unit_coordinates。
    """
    # 解析 JSON 字符串
    data = json.loads(input_json)

    # 初始化空数组来存储所有的 building_coordinates 和 unit_coordinates
    all_building_ids = []
    all_unit_ids = []
    all_building_coordinates = []
    all_unit_coordinates = []

    # 遍历 deliveries 中的每个结果，并提取 building_coordinates 和 unit_coordinates
    for delivery in data["deliveries"]:
        all_building_ids.append(delivery["building_ids"])
        all_unit_ids.append(delivery["unit_ids"])
        all_building_coordinates.append(delivery["building_coordinates"])
        all_unit_coordinates.append(delivery["unit_coordinates"])

    # 返回结果
    return all_building_ids, all_unit_ids, all_building_coordinates, all_unit_coordinates

# 示例用法
# input_json = call_gpt4o_mini_llm(user_input)
# building_coords, unit_coords = extract_coordinates(input_json)
# print(building_coords)
# print(unit_coords)

