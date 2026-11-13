# 台灣口音男聲樣本（2026-11-13）：Breeze-TTS-2（依描述設計）＋ Qwen3 VoiceDesign
import time, wave, numpy as np
import mlx.core as mx
from mlx_audio.tts.utils import load_model
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")  # 輸出到本資料夾
TEXT = "好，我查到了。台灣最高的山是玉山，大概三千九百五十二公尺，也是東北亞第一高峰喔。"

def save(name, results, t0):
    a = np.concatenate([np.array(r.audio, dtype=np.float32) for r in results])
    w = wave.open(OUT + name + ".wav", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()
    print(f"{name}: 產生 {time.time()-t0:.1f}s，長度 {len(a)/24000:.1f}s", flush=True)

DESC = [
    "一位三十五歲左右的台灣男性，標準台灣口音的國語，聲音低沉渾厚、帶磁性，語氣沉穩自然、溫暖放鬆，像深夜電台主持人。",
    "一位四十歲的台灣男性，台灣腔國語，嗓音厚實低沉、有磁性，說話不疾不徐，沉穩可靠，像值得信任的管家。",
    "一位三十歲的台灣男生，自然的台灣口音，聲音偏低、乾淨有磁性，語氣輕鬆親切，像跟好朋友聊天。",
]
m = load_model("mlx-community/Breeze-TTS-2-mlx-8bit")
for i, d in enumerate(DESC):
    mx.random.seed(7 + i)
    t0 = time.time()
    try:
        save(f"T_breeze{i+1}", list(m.generate(text=TEXT, instruct=d, cfg_scale=4)), t0)
    except Exception as e:
        print("breeze", i, "失敗", repr(e)[:300], flush=True)
t0 = time.time()
for r in m.generate(text=TEXT, instruct=DESC[0], cfg_scale=4, stream=True, streaming_interval=0.32):
    print(f"Breeze 串流第一段：{time.time()-t0:.2f}s", flush=True); break
print("Breeze 完成", flush=True)
