import os
import json

# 设置路径
beats_dir = '/Users/jackson/Codes/fancy-musicgen/finetune/data/raw-data/beats'
metadata_file = '/Users/jackson/Codes/fancy-musicgen/finetune/data/raw-data/raw-metadata.json'

# 获取beats文件夹中的所有mp3文件
mp3_files = [f for f in os.listdir(beats_dir) if f.endswith('.mp3')]
mp3_files.sort()  # 按文件名排序

# 创建metadata列表
metadata = []
for i, filename in enumerate(mp3_files, 1):
    # 从文件名中移除.mp3扩展名作为text
    text = filename[:-4]  # 移除最后的.mp3
    # audio字段按顺序编号
    audio = f'beat{i}'
    
    metadata.append({
        "audio": audio,
        "text": text
    })

# 写入metadata到JSON文件
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"已成功整理{len(mp3_files)}个音频文件的信息到{metadata_file}")
print(f"生成的metadata前5项示例：")
for i, item in enumerate(metadata[:5], 1):
    print(f"{i}. audio: {item['audio']}, text: {item['text']}")