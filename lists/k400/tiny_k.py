import random

path = '/home/linrui/moe/TimeMoE/lists/k400/'
list = ['trainlist.txt', 'vallist.txt', 'testlist.txt']
for i in list:
    with open(path + i, 'r') as f:
        lines = f.readlines()
        # 按label分类
        label_dict = {}
        for line in lines:
            line = line.strip()
            if line:
                label = int(line.split(' ')[1])
                if label not in label_dict:
                    label_dict[label] = []
                label_dict[label].append(line)
    
    # 每个类别随机选取10%的视频
    selected_lines = []
    for label in range(400):
        if label in label_dict:
            videos = label_dict[label]
            sample_count = max(1, int(len(videos) * 0.1))
            selected_lines.extend(random.sample(videos, sample_count))
    
    # 写入新文件
    output_file = path + 'tiny_' + i
    with open(output_file, 'w') as out_f:
        for line in selected_lines:
            out_f.write(line + '\n')
