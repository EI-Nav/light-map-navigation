import pandas as pd
from math import sqrt

def extract_end_records(file_path):
    """从CSV文件中提取结束记录，并计算每段轨迹的长度"""
    df = pd.read_csv(file_path)

    end_records = []
    track_lengths = []
    current_track_length = 0.0
    start_index = None

    for i in range(len(df)):
        if df.at[i, 'status'] == 'start':
            if start_index is not None:  # 不是第一个start
                # 保存前一段的结束记录和轨迹长度
                end_records.append(df.iloc[i-1].to_dict())
                track_lengths.append(current_track_length)
            # 重置轨迹长度并记录新的start索引
            current_track_length = 0.0
            start_index = i
        elif start_index is not None:  # 计算从当前start开始的轨迹长度
            x1, y1 = df.at[i-1, 'robot_x'], df.at[i-1, 'robot_y']
            x2, y2 = df.at[i, 'robot_x'], df.at[i, 'robot_y']
            current_track_length += sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # 处理最后一段轨迹
    if start_index is not None and df.iloc[-1]['status'] != 'start':
        end_records.append(df.iloc[-1].to_dict())
        track_lengths.append(current_track_length)

    return end_records, track_lengths

def read_coordinates(file_path):
    """从TXT文件中读取真值坐标"""
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1]
                x, y = map(float, line.split(','))
                coordinates.append((x, y))
    return coordinates

def calculate_metrics(predicted, ground_truth):
    """计算简单的误差指标，例如欧几里得距离"""
    if len(predicted) != len(ground_truth):
        raise ValueError("预测值和真值列表长度不一致！")

    errors = []
    for (pred, gt) in zip(predicted, ground_truth):
        error = sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)
        errors.append(error)
    
    return errors

def calculate_success_rate(errors, threshold=10):
    """计算成功率，error <= threshold 为成功"""
    successes = sum(1 for error in errors if error <= threshold)
    total = len(errors)
    success_rate = successes / total if total > 0 else 0
    return success_rate, successes, total

# 使用方法：将file_path替换为你的文件路径
csv_file_path = '/home/wjh/Study/light-map-navigation/src/delivery_benchmark/result/delivery_instructions.csv'
txt_file_path = '/home/wjh/Study/light-map-navigation/src/delivery_benchmark/data/delivery_instructions_gt.txt'

# 从CSV文件提取结束记录和轨迹长度
end_records, track_lengths = extract_end_records(csv_file_path)

# 从结束记录中提取预测的坐标 (robot_x, robot_y)
predicted_coordinates = [(record['robot_x'], record['robot_y']) for record in end_records]

# 从TXT文件读取真值坐标
ground_truth_coordinates = read_coordinates(txt_file_path)

# 计算误差指标
errors = calculate_metrics(predicted_coordinates, ground_truth_coordinates)

# 计算成功率
success_rate, successes, total = calculate_success_rate(errors, threshold=10)

# 打印误差、轨迹长度和成功率
for i, (error, track_length) in enumerate(zip(errors, track_lengths)):
    print(f"Record {i + 1}: Error = {error:.4f}, Track Length = {track_length:.4f}")

print(f"\nTotal Records: {total}")
print(f"Successful Records: {successes}")
print(f"Success Rate: {success_rate:.2%}")
