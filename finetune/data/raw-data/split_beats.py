import os
import subprocess
from glob import glob

# 源文件和目标文件夹路径
BEATS_DIR = os.path.abspath('./beats')
SEGMENT_DIR = os.path.abspath('./beat-segment')

os.makedirs(SEGMENT_DIR, exist_ok=True)

# 片段参数（起始秒，输出编号）
segments = [
    (20, '1'),
    (60, '2'),
]

mp3_files = sorted(glob(os.path.join(BEATS_DIR, 'beat*.mp3')), key=lambda x: int(os.path.basename(x).replace('beat','').replace('.mp3','')))

for mp3_file in mp3_files:
    base_name = os.path.splitext(os.path.basename(mp3_file))[0]  # 'beat1' etc
    beat_num = base_name.replace('beat', '')
    for start_sec, seg_id in segments:
        out_name = f'beat{beat_num}-{seg_id}.wav'
        out_path = os.path.join(SEGMENT_DIR, out_name)
        command = [
            'ffmpeg',
            '-y',
            '-ss', str(start_sec),
            '-t', '15',
            '-i', mp3_file,
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',
            out_path
        ]
        print(f'正在处理: {mp3_file} -> {out_path}')
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f'处理 {mp3_file} 时出错: {result.stderr.decode()}')
print('处理完成！')

