import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "ESD"
    # 递归扫描所有文件夹，找到指定的 .txt 文件
    txt_files = []
    for root, _, files in os.walk(in_dir):
        for file in files:
            if file.endswith(".txt") and "0011" <= file[:4] <= "0020":
                txt_files.append(os.path.join(root, file))
    
    for txt_file in tqdm(txt_files):
        with open(txt_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue  # 跳过格式不正确的行
                base_name, text, emotion = parts
                base_name = base_name.strip()
                text = text.strip()
                text = _clean_text(text, cleaners)
                # 查找对应的 .wav 文件
                wav_path = None
                for root, _, files in os.walk(in_dir):
                    for file in files:
                        if file == f"{base_name}.wav":
                            wav_path = os.path.join(root, file)
                            break
                    if wav_path:
                        break
                
                if wav_path and os.path.exists(wav_path):
                    # 创建输出文件夹
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    
                    # 加载和处理音频文件
                    wav, _ = librosa.load(wav_path, sr=sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, f"{base_name}.wav"),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    
                    # 保存转录文本
                    with open(
                        os.path.join(out_dir, speaker, f"{base_name}.lab"),
                        "w",
                    ) as f1:
                        f1.write(text)