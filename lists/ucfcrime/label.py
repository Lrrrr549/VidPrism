with open('testlist.txt', 'r') as f:
    lines = f.readlines()
# UCF-Crime 异常类别
# abnormal_categories = {
#     'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
#     'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 
#     'Stealing', 'Vandalism'}
abnormal_categories = {
    'Abuse':1, 'Arrest':2, 'Arson':3, 'Assault':4,
    'Burglary':5, 'Explosion':6, 'Fighting':7,
    'RoadAccidents':8, 'Robbery':9, 'Shooting':10,
    'Shoplifting':11, 'Stealing':12, 'Vandalism':13
}
new_lines = []
for line in lines:
    line = line.strip()
    if line:
        # 提取类别名（文件夹名）
        category = line.split('/')[0]
        
        # 判断是否为异常视频
        if category in abnormal_categories:
            label = abnormal_categories[category]  # 异常类别对应的标签
        else:
            label = 0  # normal
            
        new_lines.append(f'{line} {label}\n')
# 写入新文件
with open('testlist_labeled.txt', 'w') as f:
    f.writelines(new_lines)
print(f'已处理 {len(new_lines)} 个视频文件')
print('标签分布：')
normal_count = sum(1 for line in new_lines if line.endswith(' 0\n'))
abnormal_count = sum(1 for line in new_lines if line.endswith(' 1\n'))
print(f'Normal (0): {normal_count}')
print(f'Abnormal (1): {abnormal_count}')