import numpy as np
import os
import cv2  

base_video_dir = '/home/linrui/moe/UCF-Crime/Anomaly-Detection-Dataset/Anomaly-Videos'  
video_list_file = '/home/linrui/moe/MoTE/lists/ucfcrime/testlist_labeled.txt'
# Ground Truth 标注文件路径
gt_txt = '/home/linrui/moe/UCF-Crime/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
# 最终输出的GT文件
output_gt_file = '/home/linrui/moe/MoTE/lists/ucfcrime/gt_ucf.npy'

try:
    with open(gt_txt, 'r') as f:
        gt_lines = f.readlines()
except FileNotFoundError:
    print(f"错误：标注文件未找到于 {gt_txt}")
    exit()

gt = []
processed_videos_count = 0
found_annotations_count = 0

print("开始生成 Ground Truth 文件...")

try:
    with open(video_list_file, 'r') as f:
        video_list_lines = f.readlines()
except FileNotFoundError:
    print(f"错误：视频列表文件未找到于 {video_list_file}")
    exit()

for line in video_list_lines:
    line = line.strip()
    if not line:
        continue

    # 从行中获取相对视频路径，例如 'Abuse/Abuse028_x264.mp4'
    relative_video_path = line.split(' ')[0]

    # 构建视频文件的完整路径
    full_video_path = os.path.join(base_video_dir, relative_video_path)

    # 检查视频文件是否存在
    if not os.path.exists(full_video_path):
        print(f"警告：视频文件未找到，已跳过 -> {full_video_path}")
        continue
    
    # --- 使用 OpenCV 获取视频总帧数 ---
    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        print(f"警告：无法打开视频文件，已跳过 -> {full_video_path}")
        continue
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release() # 及时释放资源
    # ------------------------------------

    # 提取用于在gt_txt中查找的视频名，例如 'Abuse028_x264'
    video_name_for_lookup = os.path.basename(relative_video_path).replace('.mp4', '')

    # 为当前视频创建一个全零的标签向量
    gt_vec = np.zeros(total_frames, dtype=np.float32)

    # 检查视频是否为异常视频 (通常 'Normal' 视频不在标注文件中)
    # 遍历标注文件，为异常视频打上标签
    is_abnormal_found = False
    for gt_line in gt_lines:
        if video_name_for_lookup in gt_line:
            is_abnormal_found = True
            # 解析标注信息
            gt_content = gt_line.strip().split('  ')[1:-1]
            abnormal_fragment = [[int(gt_content[i]), int(gt_content[j])] for i in range(1, len(gt_content), 2) for j in range(2, len(gt_content), 2) if j == i+1]
            
            # 将异常片段标记为1.0
            if len(abnormal_fragment) != 0:
                for frag in abnormal_fragment:
                    start_frame, end_frame = frag[0], frag[1]
                    # 确保索引不越界
                    if start_frame < total_frames:
                        # end_frame可能会超过总帧数，取两者中较小的值
                        end_frame = min(end_frame, total_frames)
                        gt_vec[start_frame:end_frame] = 1.0
            break # 找到对应标注后即可退出内层循环
    
    # 将当前视频的标签向量完整地添加到总列表中
    # 注意：这里不再使用 gt_vec[:-clip_len]，因为我们不再依赖于clip/特征的概念
    gt.extend(gt_vec)
    
    processed_videos_count += 1
    if is_abnormal_found:
        found_annotations_count += 1
    
    if processed_videos_count % 50 == 0:
        print(f"已处理 {processed_videos_count}/{len(video_list_lines)} 个视频...")

# 保存最终的 Ground Truth 数组
np.save(output_gt_file, np.array(gt))

print("\n--- 任务完成 ---")
print(f"总共处理了 {processed_videos_count} 个视频。")
print(f"为其中的 {found_annotations_count} 个视频找到了并应用了异常标注。")
print(f"最终的 Ground Truth 文件已保存至: {output_gt_file}")
print(f"该GT数组的总长度（总帧数）为: {len(gt)}")