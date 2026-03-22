import os
import cv2

# --- 配置 ---
# 包含 Abuse, Arrest 等文件夹的视频根目录
base_video_dir = '/home/linrui/moe/UCF-Crime/Anomaly-Detection-Dataset/Anomaly-Videos' # <--- !!! 请务必修改这里 !!!
# 原始的列表文件 (格式: path label)
input_list_file = '/home/linrui/moe/MoTE/lists/ucfcrime/testlist_labeled.txt'
# 将要生成的新列表文件 (格式: path num_frames label)
output_list_file = '/home/linrui/moe/MoTE/lists/ucfcrime/testlist_with_frames.txt'

print("开始生成带帧数信息的列表文件...")

output_lines = []
with open(input_list_file, 'r') as f:
    lines = f.readlines()
    total_videos = len(lines)
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        parts = line.split(' ')
        relative_path = parts[0]
        label = parts[1]
        
        full_video_path = os.path.join(base_video_dir, relative_path)

        if not os.path.exists(full_video_path):
            print(f"警告: 视频文件未找到，已跳过 -> {full_video_path}")
            continue

        try:
            cap = cv2.VideoCapture(full_video_path)
            if not cap.isOpened():
                raise IOError("无法打开视频")
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # 按照 "path num_frames label" 格式构建新行
            new_line = f"{relative_path} {num_frames} {label}\n"
            output_lines.append(new_line)

        except Exception as e:
            print(f"错误: 处理视频 {full_video_path} 时出错: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{total_videos} 个视频...")

# 将结果写入新文件
with open(output_list_file, 'w') as f:
    f.writelines(output_lines)

print(f"\n任务完成！新的列表文件已保存至: {output_list_file}")