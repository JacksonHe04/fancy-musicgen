import os
import json

# 设置文件路径
metadata_file = '/Users/jackson/Codes/fancy-musicgen/finetune/data/raw-data/raw-metadata.json'

# 读取metadata文件
with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 更新metadata字段
updated_metadata = []
for item in metadata:
    # 创建新的条目，将text改为style
    new_item = {
        "audio": item["audio"],
        "style": item["text"],
        "emotion": ""
    }
    updated_metadata.append(new_item)

# 写回metadata文件
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(updated_metadata, f, ensure_ascii=False, indent=2)

print(f"已成功更新{len(updated_metadata)}个条目的字段")
print("字段更新：")
print("- 'text' 已重命名为 'style'")
print("- 新增了 'emotion' 字段（初始值为空）")
print(f"\n更新后的metadata前5项示例：")
for i, item in enumerate(updated_metadata[:5], 1):
    print(f"{i}. audio: {item['audio']}, style: {item['style']}, emotion: {item['emotion']}")