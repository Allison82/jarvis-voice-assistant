# 做賈維斯的聲音（台灣口音男聲）

賈維斯的嘴用 **Breeze-TTS-2**（`mlx-community/Breeze-TTS-2-mlx-8bit`，本機跑、台灣口音）。
為了「每句聲音一致」，做法是先產一段**參考聲音** `jarvis_ref.wav`，之後每句都拿它
當 `ref_audio`／`ref_text` 去複製音色。

> ⚠️ Breeze-TTS-2 **僅限非商業用途**。這裡的 `demo_*.ogg` 只是示範音，請遵守模型授權。

## 試聽：三個候選男聲 + 中英示範

| 檔案 | 內容 |
|------|------|
| `demo_T_breeze1.ogg` | 候選1：35 歲台灣男性，深夜電台主持人感 |
| `demo_T_breeze2.ogg` | 候選2：40 歲台灣男性，沉穩可靠管家感 |
| `demo_T_breeze3.ogg` | 候選3：30 歲台灣男生，像跟好朋友聊天（本專案採用這個當 `jarvis_ref`）|
| `demo_E_english.ogg` | 同一個聲音念純英文 |
| `demo_E_mixed.ogg` | 同一個聲音念中英夾雜（科技業口吻）|

## 自己做一版

1. 裝好 Breeze：環境裡 `pip install mlx mlx-audio`（跟主程式共用那個 venv 即可）。
2. 跑產生器（會在本資料夾輸出 `T_breeze1~3.wav`）：

   ```bash
   ~/nemotron-asr-test/.venv/bin/python make_ref_voice.py
   ```

3. 聽一聽，挑一個喜歡的，重新命名成參考音：

   ```bash
   cp T_breeze3.wav jarvis_ref.wav
   ```

4. 打開 `../jarvis.py`，確認這兩個對得起來：
   - `TTS_REF`：指到 `voices/jarvis_ref.wav`
   - `TTS_REF_TEXT`：**就是那段參考音念的那句話**（音檔和文字要一致，複製才準）。
     預設參考句是：「好，我查到了。台灣最高的山是玉山，大概三千九百五十二公尺，
     也是東北亞第一高峰喔。」若你用自己的錄音，就把這句改成你錄的內容。

## 想換成完全不同的聲音？

`make_ref_voice.py` 裡的 `DESC` 是「用文字描述你要的聲音」（Breeze 支援 instruct）。
改描述就能生出不同年齡、語氣、口音的聲音；或直接用你自己錄的一段乾淨人聲當
`jarvis_ref.wav`（搭配對應的 `TTS_REF_TEXT`），複製出來的音色會更像你指定的人。

> 本資料夾只收 `demo_*.ogg` 示範音；你自己產的 `*.wav` 被 `.gitignore` 擋著，不會上傳。
