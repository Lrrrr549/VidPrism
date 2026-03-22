#!/usr/bin/env python3
"""
Convert labels.json to CSV format similar to kinetics_400_labels.csv
"""

import json
import csv
import os

def convert_labels_to_csv():
    # 输入文件路径
    json_file = 'labels.json'
    # 输出文件路径
    csv_file = 'ssv2_labels.csv'
    
    # 检查输入文件是否存在
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found!")
        return
    
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    
    # 创建 CSV 文件
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入标题行
        writer.writerow(['id', 'name'])
        
        # 按照 ID 排序写入数据
        # 将字符串 ID 转换为整数进行排序
        sorted_items = sorted(labels_dict.items(), key=lambda x: int(x[1]))
        
        for label_name, label_id in sorted_items:
            writer.writerow([label_id, label_name])
    
    print(f"Successfully converted {json_file} to {csv_file}")
    print(f"Total labels: {len(labels_dict)}")

if __name__ == "__main__":
    convert_labels_to_csv()
