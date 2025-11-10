import os
import json

# 设置路径
beats_dir = '/Users/jackson/Codes/fancy-musicgen/finetune/data/raw-data/beats'
metadata_file = '/Users/jackson/Codes/fancy-musicgen/finetune/data/raw-data/raw-metadata.json'

# 读取metadata文件
with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 获取当前beats文件夹中的所有文件
current_files = {f[:-4]: f for f in os.listdir(beats_dir) if f.endswith('.mp3')}

# 重命名文件
renamed_count = 0
skipped_count = 0

for item in metadata:
    text = item['text']
    audio = item['audio']
    
    if text in current_files:
        old_filename = current_files[text]
        old_path = os.path.join(beats_dir, old_filename)
        new_path = os.path.join(beats_dir, f"{audio}.mp3")
        
        # 检查新文件名是否已存在
        if os.path.exists(new_path):
            print(f"跳过: {audio}.mp3 已存在")
            skipped_count += 1
        else:
            os.rename(old_path, new_path)
            print(f"重命名: {old_filename} -> {audio}.mp3")
            renamed_count += 1
    else:
        print(f"未找到: {text}.mp3")
        skipped_count += 1

print(f"\n重命名完成！")
print(f"成功重命名: {renamed_count} 个文件")
print(f"跳过: {skipped_count} 个文件")