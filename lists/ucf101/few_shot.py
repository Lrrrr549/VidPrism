import random
from collections import defaultdict

train_path = '/home/linrui/moe/MoTE/lists/ucf101/trainlist01-1.txt'
shot = 2
# Read the original train list
with open(train_path, 'r') as f:
    lines = f.readlines()

# Group videos by class
class_videos = defaultdict(list)
for line in lines:
    line = line.strip()
    if line:
        class_name = line.split('/')[0]
        class_videos[class_name].append(line)

# Sample 16 videos per class
few_shot_lines = []
for class_name, videos in class_videos.items():
    if len(videos) >= shot:
        sampled_videos = random.sample(videos, shot)
    else:
        sampled_videos = videos
    few_shot_lines.extend(sampled_videos)

# Save to new file
output_path = train_path.replace('trainlist01-1.txt', f'trainlist01-{shot}shot.txt')
with open(output_path, 'w') as f:
    for line in few_shot_lines:
        f.write(line + '\n')

print(f"16-shot training list saved to {output_path}")