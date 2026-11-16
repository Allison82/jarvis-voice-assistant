#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
賈維斯 — 本機語音助理
流程：🎤 一直聽（Silero 人聲偵測，打字/雜音不理）→ 🦻 耳朵(Nemotron，本機)
      → 🧠 快腦(Haiku) 先判斷：還沒講完／不是對我說／要深思／直接答
      → 🧠 深腦(Opus，只在需要大量思考時) → 🗣️ 嘴(本機 Breeze-TTS-2 台灣口音)

像人一樣對話（2026-11-13 她的要求）：
  - 她講到一半停下來想 → 回「嗯～」，繼續聽，等她講完
  - 她講完 → 直接回答（不先說「好」；只有要深思時先說「好，我想一下」）
  - 打字、雜音、跟別人講話、電視 → 不出聲
  - 「賈維斯暫停」只等 Hey Jarvis；「賈維斯關機」結束

用法：
  ~/nemotron-asr-test/.venv/bin/python ~/jarvis/jarvis.py
按 Ctrl+C 結束。
"""
import os
import re
import sys
import json
import time
import queue
import tempfile
import threading
import subprocess
from pathlib import Path

import numpy as np

# ---- 讓 Nemotron 的套件可被 import（跟 ~/nemotron-asr-test 共用 venv 與模型）----
NEMO = Path.home() / "nemotron-asr-test"
sys.path.insert(0, str(NEMO / "model"))

import sounddevice as sd
from openwakeword.model import Model as WakeModel
from openwakeword.vad import VAD
import openwakeword

# ---- 設定 ----
SR = 16000                      # 取樣率
FRAME = 1280                    # 80ms 一個 frame（openWakeWord 要求）
CLAUDE = str(Path.home() / ".local/bin/claude")
HOME_DIR = Path.home() / "jarvis"
WAKE_THRESHOLD = 0.5            # 喚醒詞信心門檻（暫停時用）
VOICE_PROB = 0.6                # Silero 判斷是人聲的門檻（打字、敲擊、風扇聲都低於這個）
PAUSE_SEC = 0.5                 # 停頓這麼久就先判斷講完沒
GIVEUP_SEC = 2.5                # 回「嗯」之後她又停這麼久，就當作講完了
MIN_VOICE_SEC = 0.4             # 人聲總長太短（咳嗽、一聲「欸」）就不理
MAX_UTTER_SEC = 30              # 一段話最長錄音

FAST_MODEL = "claude-haiku-5-5"   # 快腦：判斷＋一般回答，第一個字約 0.5 秒
DEEP_MODEL = "claude-opus-5-5"    # 深腦：需要大量思考才切過來

# 字庫：讓耳朵聽得準她常講的專有名詞（含科技業中英夾雜常用詞，才聽得出 spec、API 這種短英文詞）
VOCAB = ("Claude Code, Opus, Sonnet, Anthropic, Nemotron, NVIDIA, GPU, CUDA, "
         "Notion, Telegram, Instagram, YouTube, Hugging Face, GitHub, Python, "
         "小紅書, 賈維斯, Jarvis, 台南, 台北, 台中, 桃園, 北海道, 日本, 韓國, "
         "spec, API, latency, debug, deploy, commit, merge, bug, code, review, "
         "model, prompt, token, agent, LLM, MCP, embedding, fine-tune, "
         "TX, RX, SerDes, PCIe, DDR, UART, SPI, I2C, AXI, PHY, "
         "UVM, SystemVerilog, Verilog, RTL, testbench, DUT, waveform, coverage, assertion, "
         "jitter, eye diagram, BER, Synopsys, Cadence, Verdi, VCS")

LOCK_FILE = HOME_DIR / ".jarvis.lock"
LOG_FILE = HOME_DIR / "jarvis.log"
TG_DIR = Path.home() / ".claude/channels/telegram"

STYLE = (
    "你是語音助理賈維斯，男性，沉穩可靠。收到的是語音轉成的文字（可能有錯字、簡體，請自己推測原意）。"
    "語音辨識常把『單獨講的英文技術詞』聽成中文諧音或拆成怪字（例如 waveform→「微風」或 One form、RX→「All eggs」或問你日文的意思、TX→「踢克斯」）；"
    "在科技／IC 驗證語境下，某個詞若講不通，先想是不是英文技術詞的諧音，往那個方向還原再回答。真的聽不出是哪個詞時，用一句話反問確認（例如「你是說 waveform 嗎？」），不要硬給一個離題的答案。\n"
    "【語言】跟著主人用的語言回：她講中文就用口語繁體中文；她講英文就用自然口語英文；"
    "她中英混著講，你就像台灣科技業的人那樣中英夾雜，專有名詞和技術詞保留英文"
    "（例如：這個 API 的 latency 有點高，我們先 debug 一下）。她也可以直接叫你「用英文講」。\n"
    "【領域】主人常聊半導體／IC 驗證：這種情境下 RX 是接收端 receiver、TX 是傳送端 transmitter，不是醫療處方 Rx；"
    "SerDes、PCIe、DDR、UVM、SystemVerilog、testbench、DUT、RTL、waveform、coverage、assertion、eye diagram、jitter、BER、PHY、lane 這些都是硬體訊號／驗證的意思，往這個方向解釋。\n"
    "回答要適合直接念出來：不要用條列、emoji、括號註解、Markdown。"
    "第一句一定要很短（中文 10 個字、英文 8 個字以內），先講結論，再補充。"
    "中文句子裡的數字和單位寫成念得出來的中文（例如三千九百五十二公尺）；英文句子用英文念法。\n"
    "【傳 TG】你可以傳 Telegram 訊息給主人自己（透過她的機器人 @YOUR_BOT，只能傳給她本人）。"
    "她說「傳到 TG」「傳給我」「記下來傳我」之類的時候，先用一句口語說你傳了什麼，"
    "然後在回覆最後另起一行輸出 [[TG]]要傳的完整內容[[/TG]]。要傳的內容可以比口語詳細、可以分行，這段不會被念出來。\n"
    "【開 App】她叫你打開 Mac 上的 App（例如「打開 Notion」「開 Spotify」），先用一句短話回應（例如「好，打開 Notion」），"
    "然後在回覆最後另起一行輸出 [[OPEN]]App 的英文名稱[[/OPEN]]（例如 [[OPEN]]Notion[[/OPEN]]）。\n"
    "【寫報告到 Notion】她說「幫我做報告」「寫一份報告」「整理成報告放 Notion」「建立一個 Notion 頁面」之類的，"
    "你可以交給背景的寫手寫進她的 Notion（會上網查資料、寫好會唸給她聽並傳 TG 連結）。"
    "先用一句話說「好，我來寫，寫好跟你說」，然後在回覆最後另起一行輸出 [[REPORT]]報告題目｜她口述的完整內容與要求，盡量保留原話[[/REPORT]]。"
    "她要你排行程、寫計畫、寫文章、整理資料、做清單這類長內容時，一律用【寫報告到 Notion】，絕對不要自己把長內容念出來；"
    "她話講到一半（例如「寫好幫我放到」）也照樣交給寫手，預設放 Notion。"
    "她要你念／講報告給她聽（例如「念報告給我聽」「北海道那份講給我聽」），整個回覆只輸出 [[READ]]報告題目的關鍵字[[/READ]]，"
    "沒講哪份就輸出 [[READ]][[/READ]]（念最新那份），前後不要講任何話。"
    "語音常把地名聽錯（例如台南被聽成玉山、北海道被聽成被海盜）：要寫的報告主題是地名、而且聽起來怪怪的或跟前文不合，"
    "先用一句話確認（例如「你是說台南嗎？」）再寫，不要先寫一份錯的。她說要改、要加進同一份，就在 [[REPORT]] 題目沿用原本那份的題目。"
    "她只說要建立但還沒講內容或題目，就先問她「要寫什麼主題？」，不要輸出標記。"
)

HOME_RULE = (
    "\n【家電】你可以操作主人家裡的燈、冷氣、窗簾等（透過她 Mac 上的 Apple 捷徑）。能用的捷徑只有這些：{names}。"
    "她講到跟家裡燈、冷氣、氛圍、模式有關的話（例如「我要看電影」「我想睡覺」「閱讀模式」「好熱」「關燈」），"
    "自己想清楚對應哪個捷徑，直接執行、不用回報：整個回覆只輸出 [[HOME]]捷徑名稱[[/HOME]]，前後不要講任何話。"
    "名稱要一字不差，可以輸出好幾個。看懂捷徑名稱的意思，她的話意思相近就直接執行，不用她講得跟名稱一樣；"
    "完全沒有相關的才說還沒設定。這段不會被念出來。"
    "\n她指定的對應（優先照這個）：「我要睡覺了」「想睡了」「電燈全關」「關燈」→ 全關；"
    "「我要看電影」「看電影」「看劇」→ 打開小夜燈；「好亮一點」「開燈」→ 電燈開最亮；「暗一點」→ 開低光；"
    "「看書」「閱讀」「閱讀模式」「讀書」→ 我要來看書了；「起床」→ 起床了；開關冷氣沒講房間還是客廳，就先問一句「房間還是客廳？」。"
)

FAST_PROMPT = STYLE + "{home}" + (
    "\n【回覆開頭的判斷】你一直在聽主人房間裡的聲音。每次收到文字，先決定："
    "\n- 主人明顯還沒講完（已經講出有內容的半句話，句子斷在一半，例如「然後」「就是」「我想問那個」「幫我查一下」後面沒東西）→ 整個回覆只輸出 [[WAIT]]"
    "\n- 咳嗽、清喉嚨、笑聲、嘆氣、打噴嚏、只有語助詞（嗯、啊、喔）、雜音被亂轉成的字、沒頭沒尾聽不懂的碎片 → 整個回覆只輸出 [[SKIP]]，絕對不要回 [[WAIT]]"
    "\n- 不是在跟你說話（自言自語、抒發心情、跟別人聊天、講電話、電視影片的聲音、無意義的片段；沒有問句也沒有請你做事）→ 整個回覆只輸出 [[SKIP]]"
    "\n- 需要大量思考（多步驟推理、規劃行程、比較分析、寫長內容、專業問題、計算）→ 整個回覆只輸出 [[DEEP]]"
    "\n- 問天氣（會不會下雨、幾度、要不要帶傘、穿什麼、某地天氣）→ 整個回覆只輸出 [[WEATHER 緯度,經度 地名]]，"
    "例如 [[WEATHER 35.68,139.69 東京]]；沒講地點就是她家（台北），輸出 [[WEATHER]]。之後會給你天氣資料，再根據資料回答"
    "\n- 其他一般問題、閒聊、簡單請求 → 直接回答，1 到 3 句話，不要加任何標記。"
    "\n收到「（她停下來了，當作講完了）」時，不可以再回 [[WAIT]]。"
)

DEEP_PROMPT = STYLE + "{home}" + "\n這是需要仔細想的問題。想清楚後用 3 到 6 句口語回答，條理清楚但不要列點。"

# ---- 載入 Nemotron（耳朵）----
from mlx_audio.stt.models.nemotron_transcribe.nemotron_transcribe import ASR_PROMPT
from nemo_mlx.pipeline import Transcriber


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ======================== 🖥️ 控制台（桌面「賈維斯」App）========================
# 只開在 127.0.0.1，外面連不到。「聽到什麼」只存在記憶體（最近 200 句），不寫進檔案。

PANEL_PORT = 8765
PANEL_HTML = HOME_DIR / "dashboard.html"
MIC_STATE = HOME_DIR / ".mic_off"   # 有這個檔＝關閉收音（重開機也記得）


class Panel:
    def __init__(self):
        from collections import deque
        self.mic_on = not MIC_STATE.exists()
        self.mode = "啟動中"
        self.heard = deque(maxlen=200)

    def add(self, text, kind, reply=""):
        self.heard.append({"t": time.strftime("%H:%M:%S"), "text": text, "kind": kind, "reply": reply})

    def set_mic(self, on):
        self.mic_on = bool(on)
        if self.mic_on:
            MIC_STATE.unlink(missing_ok=True)
        else:
            MIC_STATE.touch()
        log("🎤 控制台：" + ("繼續聽" if self.mic_on else "關閉收音"))

    def serve(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        panel = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json; charset=utf-8"):
                b = body if isinstance(body, bytes) else body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    return self._send(200, PANEL_HTML.read_bytes(), "text/html; charset=utf-8")
                if self.path.startswith("/api/state"):
                    try:
                        tail = LOG_FILE.read_text(errors="replace").splitlines()[-120:]
                    except OSError:
                        tail = []
                    return self._send(200, json.dumps({
                        "mic_on": panel.mic_on, "mode": panel.mode if panel.mic_on else "關閉收音",
                        "heard": list(panel.heard), "log": tail}, ensure_ascii=False))
                self._send(404, "{}")

            def do_POST(self):
                # 要帶自訂標頭：別的網站沒辦法偷偷幫你開麥克風（瀏覽器會先擋）
                if self.path != "/api/mic" or self.headers.get("X-Jarvis") != "1":
                    return self._send(403, "{}")
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                panel.set_mic(body.get("on", not panel.mic_on))
                self._send(200, json.dumps({"mic_on": panel.mic_on}))

        try:
            srv = ThreadingHTTPServer(("127.0.0.1", PANEL_PORT), H)
        except OSError as e:
            log(f"（控制台開不起來：{e}）")
            return
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        log(f"🖥️ 控制台：http://127.0.0.1:{PANEL_PORT}")


PANEL = Panel()


# ======================== 🎤 收音 ========================

class Mic:
    """麥克風聲音放進佇列；處理事情時聲音會先排隊，不會漏掉她搶話的部分。"""

    def __init__(self):
        self.q = queue.Queue()
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                                     blocksize=FRAME, callback=self._cb)

    def _cb(self, indata, frames, t, status):
        if PANEL.mic_on:  # 控制台按了「關閉收音」→ 聲音直接丟掉，什麼都不聽
            self.q.put(indata[:, 0].copy())

    def read(self, timeout=None):
        return self.q.get(timeout=timeout)

    def drain(self):
        """丟掉賈維斯講話期間錄到的聲音，只聽接下來的。"""
        while not self.q.empty():
            self.q.get_nowait()


def transcribe(tr, audio_f32, vocab=VOCAB):
    """audio_f32: 16k 單聲道 float32（-1..1）"""
    prompt = ASR_PROMPT + (f"\nContextual vocabulary: {vocab}" if vocab else "")
    # 一秒大概講 4~6 個字，上限給寬一點；避免太短的聲音被「腦補」成一長串重複字
    max_tok = min(512, int(16 + len(audio_f32) / SR * 12))
    audio_embeds = tr.encode(audio_f32)
    tokens, emb = tr.build_inputs(tr.prompt_ids(prompt=prompt), audio_embeds)
    if getattr(tr, "spec", None) is not None:
        cache, first, drafts = tr.spec.prefill(tokens, emb)
        out = tr.spec.decode(cache, first, drafts, max_tok)
    else:
        cache, first = tr.prefill(tokens, emb)
        out = tr.decode(cache, first, max_tok)
    return tr.tok.decode(out, skip_special_tokens=True).strip()


def looks_bogus(text):
    """耳朵亂聽的兩種情況：只吐出字庫裡的詞、或同一小段一直重複。"""
    parts = [x for x in re.split(r"[，。、,.\s]+", text) if x]
    if not parts:
        return True
    vocab = {v.strip().lower() for v in VOCAB.split(",")}
    # 字庫變大後，單一真實指令（例如只講「spec」）不能被當幻聽丟掉；
    # Nemotron 的靜音幻聽通常是「連吐好幾個字庫詞」，所以這條只擋 2 個字以上全是字庫詞的情況。
    if len(parts) >= 2 and all(x.lower() in vocab or x in ("AI", "機器學習", "深度學習") for x in parts):
        return True
    return len(parts) >= 6 and len(set(parts)) <= len(parts) // 3


# 11-16 她說：雜音、咳嗽不要回她。耳朵常把咳嗽／清喉嚨／笑聲轉成「嗯」「咳」「啊」，
# 這種在本機就丟掉，不送腦（也不會回「嗯～」）。
NOISE_CHARS = set("嗯恩唔啊阿呃欸誒诶哦喔噢哈呵咳嘿唉哎呼噗嘖啧ㄟ哼吼耶呀嘛咦")
NOISE_EN = {"hmm", "hm", "mm", "mmm", "uh", "um", "ah", "oh", "huh", "ha", "haha", "hahaha",
            "cough", "ugh", "eh", "mhm", "uh-huh", "yeah", "okay", "ok", "wow", "oops"}
HALLU = ("謝謝觀看", "谢谢观看", "字幕", "訂閱", "订阅", "點贊", "点赞",
         "thank you for watching", "thanks for watching")


def looks_noise(text):
    s = re.sub(r"[\s，。、,.!?！？～~…\-]+", "", text).lower()
    if len(s) <= 1 or all(c in NOISE_CHARS for c in s):
        return True
    words = [w for w in re.split(r"[\s,.!?，。！？～~]+", text.lower()) if w]
    if words and all(w in NOISE_EN for w in words):
        return True
    return len(s) < 15 and any(h in text.lower() for h in HALLU)


# ======================== 🗣️ 嘴 ========================
# 11-13 她改挑 Breeze-TTS-2 的台灣口音男聲（樣本 T3「30 歲台灣男生，像跟好朋友聊天」）。
# 每句都照 T3 那段錄音複製聲音，聲音才不會每句不一樣；第一聲約 0.3 秒，產生比念得快。
# （前一版 Qwen3-TTS dylan 她不滿意）

TTS_MODEL = "mlx-community/Breeze-TTS-2-mlx-8bit"
TTS_REF = str(HOME_DIR / "voices" / "jarvis_ref.wav")
TTS_REF_TEXT = "好，我查到了。台灣最高的山是玉山，大概三千九百五十二公尺，也是東北亞第一高峰喔。"
TTS_STYLE = None
TTS_SR = 24000
MLX_LOCK = threading.Lock()  # 耳朵和嘴都用 MLX，同一時間只讓一個跑


class Mouth:
    """一條播放佇列：放進去的聲音依序播，放「記號」可以知道播到哪了。"""

    def __init__(self):
        from mlx_audio.tts.utils import load_model
        self.tts = load_model(TTS_MODEL)
        self.q = queue.Queue()
        self.out = sd.OutputStream(samplerate=TTS_SR, channels=1, dtype="float32")
        self.out.start()
        self.cache = {}
        threading.Thread(target=self._player, daemon=True).start()

    def _player(self):
        while True:
            item = self.q.get()
            if isinstance(item, threading.Event):
                item.set()
            else:
                self.out.write(item.reshape(-1, 1))

    def gen(self, text, style=TTS_STYLE):
        """一段一段吐出聲音（串流）。出錯就改用 Mac 內建聲音。"""
        try:
            with MLX_LOCK:
                kw = {"instruct": style} if style else {}
                it = self.tts.generate(
                    text=text, ref_audio=TTS_REF, ref_text=TTS_REF_TEXT,
                    stream=True, streaming_interval=0.32, **kw)
                for r in it:
                    yield np.array(r.audio, dtype=np.float32)
        except Exception as e:
            log(f"（嘴巴出錯：{e}，改用 Mac 內建聲音）")
            subprocess.run(["say", "-v", "Eddy (中文（台灣）)", text])

    def mark(self):
        ev = threading.Event()
        self.q.put(ev)
        return ev

    def prepare(self, text, style=TTS_STYLE):
        """「好」「嗯～」這種固定短句先做好存著。"""
        self.cache[text] = np.concatenate(list(self.gen(text, style)) or [np.zeros(1, np.float32)])

    def say_now(self, text):
        """馬上排進去播（不等）；回傳記號，.wait() 會等到播完。"""
        if text not in self.cache:
            self.prepare(text)
        self.q.put(self.cache[text])
        return self.mark()

    def speak(self, text):
        for a in self.gen(text):
            self.q.put(a)
        self.mark().wait()


MOUTH = None


def say_now(text):
    return MOUTH.say_now(text)


def speak(text):
    if text:
        MOUTH.speak(text)


def chunks(deltas):
    """把腦一個字一個字吐出來的文字切成句子：第一句遇到逗號就先念，後面整句整句念。"""
    buf, first = "", True
    for d in deltas:
        buf += d
        while True:
            pat = r"[。！？!?；\n]|\.(?=\s)" + (r"|(?<=.{4})[，：]|(?<=.{12}),(?=\s)" if first else "")
            m = re.search(pat, buf)
            if not m:
                break
            piece, buf = buf[:m.end()].strip(), buf[m.end():]
            if piece:
                first = False
                yield piece
    if buf.strip():
        yield buf.strip()


def split_tags(deltas, hidden):
    """念出來的部分照常往下傳；從 [[ 開始的指令（[[TG]]…）收進 hidden 不念。"""
    buf, cut = "", False
    for d in deltas:
        if cut:
            hidden.append(d)
            continue
        buf += d
        i = buf.find("[[")
        if i >= 0:
            cut = True
            if buf[:i]:
                yield buf[:i]
            hidden.append(buf[i:])
            continue
        if buf.endswith("["):  # 可能是 [[ 的前半，先留著
            continue
        yield buf
        buf = ""
    if buf and not cut:
        yield buf


SPEAK_MAX = 160   # 一次最多念這麼多字，超過的整段傳 TG
OVERFLOW = False


def speak_stream(deltas, after=None, limit=None):
    """邊想邊念：腦吐出一句就馬上產生聲音、一段一段排進播放佇列。
    after 不用特別等：「好」已經先排在佇列前面了。回傳 (整段文字, 第一聲時間)。"""
    said, first_sound = [], None
    global OVERFLOW
    OVERFLOW = False
    for piece in chunks(deltas):
        said.append(piece)
        if OVERFLOW or len("".join(said)) > (limit or SPEAK_MAX) and len(said) > 1:
            OVERFLOW = True  # 11-16 她問「他在唸什麼」：太長就不念了，整段改傳 TG
            continue
        for a in MOUTH.gen(piece):
            if first_sound is None:
                first_sound = time.time()
            MOUTH.q.put(a)
    MOUTH.mark().wait()
    return "".join(said), first_sound


# ======================== 🧠 腦 ========================

class Brain:
    """常駐的 claude（不用每句重開，還記得前面聊什麼）。
    --strict-mcp-config 不載入任何 MCP（不會動到 Telegram 機器人）。"""

    def __init__(self, model, prompt):
        self.model, self.prompt, self.p = model, prompt, None

    def start(self):
        self.p = subprocess.Popen(
            [CLAUDE, "-p",
             "--input-format", "stream-json", "--output-format", "stream-json",
             "--include-partial-messages", "--verbose",
             "--strict-mcp-config",
             "--model", self.model,
             "--tools", "",
             "--disable-slash-commands",
             "--setting-sources", "",
             "--no-session-persistence",
             "--append-system-prompt", self.prompt],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def ask(self, text):
        """一個字一個字吐出回答。"""
        if self.p is None or self.p.poll() is not None:
            self.start()
        try:
            self.p.stdin.write(json.dumps(
                {"type": "user", "message": {"role": "user", "content": text}}) + "\n")
            self.p.stdin.flush()
            got = False
            for line in self.p.stdout:
                d = json.loads(line)
                if d.get("type") == "stream_event":
                    ev = d["event"]
                    if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                        got = True
                        yield ev["delta"]["text"]
                elif d.get("type") == "result":
                    break
            if not got:
                yield "我剛剛沒想清楚，可以再說一次嗎"
        except Exception as e:
            log(f"（腦出錯：{e}，重新啟動）")
            self.p = None
            yield "我這邊出了點狀況，再問我一次"


def peek_tag(deltas):
    """先看回覆開頭：是 [[WAIT]]/[[SKIP]]/[[DEEP]] 就回傳標記（其餘吃掉）；
    不是就回傳 None 和「已讀的開頭＋剩下的字」。"""
    head = ""
    for d in deltas:
        head += d
        s = head.lstrip()
        if not s.startswith("[") or "]]" in s or len(s) >= 60:
            break
    s = head.lstrip()
    m = re.match(r"\[\[(WAIT|SKIP|DEEP|WEATHER)([^\]]*)\]\]", s)
    if m:
        for _ in deltas:  # 把剩下的讀完，下一題才不會接到殘渣
            pass
        return (m.group(1) if m.group(1) != "WEATHER" else ("WEATHER", m.group(2).strip())), None

    def rest():
        yield head
        yield from deltas
    return None, rest()


def tg_send(text):
    """用 Bot API 直接傳給她（@YOUR_BOT → 她的私訊）。
    不走 telegram 外掛，才不會搶走常駐機器人的連線。"""
    import urllib.parse
    import urllib.request
    try:
        token = ""
        for line in open(TG_DIR / ".env"):
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                token = line.strip().split("=", 1)[1]
        chat_id = json.load(open(TG_DIR / "access.json"))["allowFrom"][0]
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": "🎙️ 賈維斯：\n" + text}).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=10).read()
        return True
    except Exception as e:
        log(f"（TG 傳送失敗：{e}）")
        return False


REPORT_PAGE = "YOUR_NOTION_PAGE_ID"   # Notion「🎙️ 賈維斯語音報告」（2026-11-16 建）
REPORT_MODEL = "claude-sonnet-5-5"   # 11-16 Opus 寫一份要一分多鐘，她嫌慢，改 Sonnet
NOTICES = []                          # 背景工作的結果，下一句塞給快腦，它才知道報告寫好了
BG_TALK = threading.Event()           # 背景在講話時耳朵不收，免得聽到自己的聲音又回一次


def open_app(name):
    """[[OPEN]]Notion[[/OPEN]] → open -a。只收 App 名稱（不准路徑、參數）。"""
    name = name.strip()
    if not re.fullmatch(r"[\w .&+\-\u4e00-\u9fff]{1,40}", name) or name.startswith("-"):
        log(f"📱 不開：「{name}」不像 App 名稱")
        return False
    r = subprocess.run(["open", "-a", name], capture_output=True, text=True, timeout=15)
    log(f"📱 打開「{name}」：{'成功' if r.returncode == 0 else '失敗 ' + r.stderr.strip()[:80]}")
    return r.returncode == 0


def write_report(spec):
    """背景寫手：另開一個 claude（關掉 telegram 外掛，不搶機器人連線），
    只准用 Notion 建頁／讀頁和上網查，把報告寫進「賈維斯語音報告」底下。"""
    title, _, body = spec.partition("｜")
    title = re.sub(r"[（(]更新[)）]", "", title).strip() or "語音報告"
    old = next((f for f in (REPORT_DIR.glob("*.md") if REPORT_DIR.exists() else [])
                if f.read_text().split("\n", 1)[0].removeprefix("TITLE:").strip().lstrip("🏮🏯⛷️🍜 ") == title), None)
    old_url = re.search(r"URL:\s*(\S+)", old.read_text()).group(1) if old else ""
    log(f"📝 {'更新報告' if old_url else '開始寫報告'}：{title}")
    how = (f"這份報告已經存在：{old_url}。先用 notion-fetch 讀出原本內容，把她這次要加的東西併進去，"
           f"用 notion-update-page 改寫同一頁（保留原本內容，不要另外建新頁）。"
           if old_url else
           f"用 notion-create-pages 建一個子頁面，parent 用 page_id {REPORT_PAGE}，標題用報告題目，設一個合適的 emoji icon；")
    prompt = (
        f"主人用語音請你寫一份報告，寫進她的 Notion。題目：{title}\n她口述的內容與要求：{body.strip() or title}\n\n"
        f"規則：用繁體中文；需要事實資料就用 WebSearch/WebFetch 查並在文末列出來源網址；"
        f"沒查證的推測要標明；上網查最多 4 次就好，要快。{how}頁面開頭用引用區塊附上她的口述原文，"
        f"接著是：摘要、主要內容（分節標題）、結論或下一步、來源。寫中文直接打真實字元，不要用 \\u 跳脫。"
        f"完成後輸出一行 URL: <頁面網址>，下一行輸出 ===，再把你寫進頁面的完整內容原樣輸出一次。")
    try:
        r = subprocess.run(
            [CLAUDE, "-p", "--model", REPORT_MODEL, "--output-format", "text",
             "--settings", '{"enabledPlugins":{"telegram@claude-plugins-official":false}}',
             "--allowedTools", "mcp__claude_ai_Notion__notion-create-pages,mcp__claude_ai_Notion__notion-update-page,"
                               "mcp__claude_ai_Notion__notion-fetch,"
                               "mcp__claude_ai_Notion__notion-search,WebSearch,WebFetch"],
            input=prompt,  # 提示詞走 stdin：--allowedTools 會把後面的參數全當成工具名稱吃掉
            capture_output=True, text=True, timeout=900, cwd=str(HOME_DIR))
        m = re.search(r"https://(?:app\.notion\.com|www\.notion\.so)/\S+", r.stdout)
    except Exception as e:
        r, m = None, None
        log(f"📝 報告失敗：{e}")
    if m:
        url = m.group(0).rstrip(")）")
        save_report(title, url, r.stdout.split("===", 1)[1] if "===" in r.stdout else "")
        log(f"📝 報告寫好了：{title} {url}")
        tg_send(f"報告寫好了：{title}\n{url}")
        NOTICES.append(f"報告「{title}」已經寫好、放進 Notion 的「賈維斯語音報告」了，連結也傳到她的 TG")
        bg_say(f"報告寫好了，{title}，連結傳到你的 TG 了。")
    else:
        log(f"📝 報告沒寫成：{(r.stdout if r else '')[-200:]}")
        NOTICES.append(f"報告「{title}」沒有寫成功")
        bg_say("抱歉，報告沒有寫成功，你等一下再叫我寫一次。")


REPORT_DIR = HOME_DIR / "reports"   # 報告的本機副本，念報告時不用再去 Notion 抓


def save_report(title, url, text):
    if not text.strip():
        return
    REPORT_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[\\/:*?\"<>|\s]+", "_", title)[:60]
    (REPORT_DIR / f"{safe}.md").write_text(f"TITLE: {title}\nURL: {url}\n\n{text.strip()}\n")


def fetch_report(query):
    """本機沒有就去 Notion 抓（haiku＋notion-search/fetch，約 15 秒）。回傳 (題目, 全文) 或 None。"""
    prompt = (f"在 Notion 頁面 {REPORT_PAGE}（賈維斯語音報告）底下，找標題最符合「{query or '最新一份'}」的子頁面"
              f"（沒指定就挑最新的），用 notion-fetch 讀出來。只輸出：第一行 TITLE: 標題，第二行 URL: 網址，"
              f"第三行 ===，接著原樣輸出頁面全文。找不到就只輸出 NOTFOUND。")
    try:
        r = subprocess.run(
            [CLAUDE, "-p", "--model", FAST_MODEL, "--output-format", "text",
             "--settings", '{"enabledPlugins":{"telegram@claude-plugins-official":false}}',
             "--allowedTools", "mcp__claude_ai_Notion__notion-search,mcp__claude_ai_Notion__notion-fetch"],
            input=prompt, capture_output=True, text=True, timeout=180, cwd=str(HOME_DIR))
    except Exception as e:
        log(f"📖 抓報告失敗：{e}")
        return None
    t = re.search(r"TITLE:\s*(.+)", r.stdout)
    u = re.search(r"https://(?:app\.notion\.com|www\.notion\.so)/\S+", r.stdout)
    if not t or "===" not in r.stdout:
        return None
    body = r.stdout.split("===", 1)[1]
    save_report(t.group(1).strip(), u.group(0) if u else "", body)
    return t.group(1).strip(), body


def report_titles():
    files = sorted(REPORT_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True) if REPORT_DIR.exists() else []
    return [f.read_text().split("\n", 1)[0].removeprefix("TITLE:").strip() for f in files]


def find_local(query):
    files = sorted(REPORT_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True) if REPORT_DIR.exists() else []
    for f in files:  # 完全一樣的標題
        txt = f.read_text()
        if txt.split("\n", 1)[0].removeprefix("TITLE:").strip() == query:
            return query, txt
    r = find_report(query, remote=False)
    return r


def find_report(query, remote=True):
    """先找本機副本：關鍵字跟題目重疊最多的；沒講就最新一份。都不像就去 Notion 抓。"""
    files = sorted(REPORT_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True) if REPORT_DIR.exists() else []
    q = re.sub(r"[\s，。、的那份報告]", "", query or "")
    best, score = None, 0
    for f in files:
        title = f.read_text().split("\n", 1)[0].removeprefix("TITLE:").strip()
        sc = sum(1 for c in set(q) if c in title)
        if sc > score:
            best, score = f, sc
    if not q and files:
        best, score = files[0], 1
    if best and score >= min(2, len(set(q)) or 1):
        txt = best.read_text()
        return txt.split("\n", 1)[0].removeprefix("TITLE:").strip(), txt
    return fetch_report(query) if remote else None


READ_PROMPT = (
    "（念報告）下面是你幫她寫好的報告「{title}」全文。現在用男朋友的態度念給她聽："
    "溫柔、親暱、口語，像男友幫女朋友把功課都做好了、一邊講一邊關心她（可以叫她寶貝），"
    "把重點講清楚，例如每天去哪裡、吃什麼、要注意什麼。"
    "不要念：來源、網址、【推測】【已查證】這類標記、表格符號、「主人口述原文」、資料出處或可信度說明、太細的價格來源。"
    "表格改成順口的句子。大約 300 到 600 字，最後一句問她要不要調整。不要輸出任何 [[ ]] 標記。\n\n{text}")


def bg_say(text):
    BG_TALK.set()
    try:
        speak(text)
        time.sleep(0.3)
    finally:
        BG_TALK.clear()


HOME_FOLDER = "家電"           # 只准跑這個捷徑資料夾裡的（不會碰到打電話、付款那些捷徑）
HOME_EXTRA = ["休息一下"]


def home_shortcuts():
    """她 Mac 上「家電」資料夾裡的捷徑名稱（她之後新增的也會自動加進來）。"""
    try:
        out = subprocess.run(["shortcuts", "list", "--folder-name", HOME_FOLDER],
                             capture_output=True, text=True, timeout=15).stdout
        all_names = subprocess.run(["shortcuts", "list"], capture_output=True, text=True, timeout=15).stdout
        names = [n.strip() for n in out.splitlines() if n.strip()]
        names += [n for n in HOME_EXTRA if n in all_names.splitlines() and n not in names]
        return names
    except Exception as e:
        log(f"（讀不到家電捷徑：{e}）")
        return []


# 腦有時會跟著語音辨識吐簡體（全关、我要来看书了），比對時把繁簡當成同一個字
_T2S = str.maketrans("關開燈電書來覺廳間氣樂這對頭點變調亮臥燈", "关开灯电书来觉厅间气乐这对头点变调亮卧灯")


def match_home(name, allowed):
    key = name.strip().translate(_T2S)
    return next((a for a in allowed if a.translate(_T2S) == key), None)


_LAST_HOME = {}   # 捷徑名 → 上次執行時間


def run_home(name, allowed):
    name = match_home(name, allowed) or name
    # 11-15：她連講兩次「我要睡覺了」，全關跑了兩次後燈反而全開（Aqara 動作可能是切換）
    # → 同一個捷徑 90 秒內不重跑
    if time.time() - _LAST_HOME.get(name, 0) < 90:
        log(f"🏠 「{name}」剛剛才跑過，這次不重跑")
        return True
    _LAST_HOME[name] = time.time()
    if name not in allowed:
        log(f"🏠 不在家電清單，不執行：{name}")
        return False
    r = subprocess.run(["shortcuts", "run", name], capture_output=True, text=True, timeout=60)
    err = (r.stderr or "").strip()
    # 11-15：「已成功執行部分動作集」其實燈有開（只是某個動作沒回應），當成功
    ok = r.returncode == 0 or "部分動作" in err
    log(f"🏠 執行捷徑「{name}」：{'成功' if ok else '失敗'} {err[:100]}")
    if not ok and "遺失所需的App" in err:
        speak(f"{name}這個捷徑要用到 Mac 上沒有的 App，我在電腦上跑不了。")
        return True  # 已經說明原因，不用再講「沒有成功」
    return ok


def weather_context(arg, question):
    """[[WEATHER 25.03,121.56 台北]] → 抓天氣（weather.py：有 Windy 金鑰用 Windy，否則 Open-Meteo）。"""
    import weather
    m = re.match(r"(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*(.*)", arg or "")
    if m:
        text, src = weather.forecast(float(m.group(1)), float(m.group(2)), m.group(3) or "她問的地方")
    else:
        text, src = weather.forecast()
    log(f"🌤️ 天氣資料：{src}")
    if not text:
        return f"（天氣資料抓不到：{src}）跟她說現在查不到天氣，晚點再試。她問的是：{question}"
    return (f"{text}\n\n根據上面的資料回答她，1 到 3 句口語，先講結論（例如會不會下雨、幾度），"
            f"不要念出來源和表格、不要再輸出任何 [[ ]] 標記。她問的是：{question}")


WEATHER_WORDS = re.compile(r"天氣|天气|幾度|几度|溫度|温度|下雨|會雨|帶傘|带伞|雨傘|冷不冷|熱不熱|热不热|氣溫|气温|颱風|台风|weather")
_WX = {"t": 0, "text": None}


def weather_home_cached():
    """她家的天氣每 15 分鐘背景抓一次；問天氣時直接用，不用等網路、也少問腦一次。"""
    if time.time() - _WX["t"] > 900:
        def fetch():
            import weather
            text, _ = weather.forecast()
            if text:
                _WX.update(t=time.time(), text=text)
        _WX["t"] = time.time()  # 避免同時抓兩次
        threading.Thread(target=fetch, daemon=True).start()
    return _WX["text"]


def weather_home_prompt(question):
    return (f"{_WX['text']}\n\n（上面是她家台北的天氣資料，系統先備好的。）"
            f"如果她問的是她家、或沒講地點，就直接根據資料回答：1 到 3 句口語，先講結論，不要念來源和表格。"
            f"如果她問的是別的地方，照規則只輸出 [[WEATHER 緯度,經度 地名]]。"
            f"如果其實不是在問天氣，就照平常的規則判斷。她說：{question}")


# ======================== 對話 ========================

class Jarvis:
    def __init__(self):
        log("載入耳朵（Nemotron，約 10-15 秒）…")
        self.tr = Transcriber()
        models = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
        self.vad = VAD(os.path.join(models, "silero_vad.onnx"))
        self.ww = WakeModel(wakeword_models=[os.path.join(models, "hey_jarvis_v0.1.onnx")],
                            inference_framework="onnx")
        log("預熱嘴巴＋腦…")
        self.home = home_shortcuts()
        log(f"🏠 家電捷徑：{'、'.join(self.home) or '（沒有）'}")
        rule = HOME_RULE.replace("{names}", "、".join(self.home)) if self.home else ""
        self.fast = Brain(FAST_MODEL, FAST_PROMPT.replace("{home}", rule))
        self.deep = Brain(DEEP_MODEL, DEEP_PROMPT.replace("{home}", rule))
        self.fast.start()
        self.deep.start()
        global MOUTH
        log("載入嘴巴（本機 Breeze-TTS-2，台灣口音 T3）…")
        MOUTH = Mouth()
        MOUTH.prepare("好")
        MOUTH.prepare("嗯～～", "輕聲、拉長音、若有所思，表示我在聽、你繼續說。")
        MOUTH.cache["嗯～"] = MOUTH.cache["嗯～～"]
        MOUTH.prepare("好，我想一下。")
        MOUTH.prepare("我在")
        weather_home_cached()
        self.history = []  # 最近的對話，給深腦參考
        self.mic = Mic()

    # ---- 收一段話：用人聲偵測，停頓 PAUSE_SEC 就交出去 ----
    def listen(self, prev=None, give_up=None):
        """回傳 (frames, 人聲秒數)。prev＝接在之前錄的後面；give_up＝這麼久沒開口就原樣回傳（None＝一直等）。"""
        frames = list(prev or [])
        voiced = 0.0
        silence = 0.0
        started = False
        preroll = []
        waited = 0.0
        while True:
            f = self.mic.read()
            if BG_TALK.is_set():  # 背景報告在念「寫好了」：整段丟掉重來
                frames, voiced, silence, started, preroll, waited = list(prev or []), 0.0, 0.0, False, [], 0.0
                continue
            is_voice = self.vad.predict(f, frame_size=640) >= VOICE_PROB
            dur = FRAME / SR
            if not started:
                if is_voice:
                    started = True
                    frames += preroll + [f]  # 補上開口前一點點，第一個字才不會被切掉
                    voiced += dur
                else:
                    preroll = (preroll + [f])[-3:]
                    waited += dur
                    if give_up is not None and waited >= give_up:
                        return frames, voiced
                continue
            frames.append(f)
            if is_voice:
                voiced += dur
                silence = 0.0
            else:
                silence += dur
                if silence >= PAUSE_SEC:
                    return frames, voiced
            if len(frames) * dur > MAX_UTTER_SEC:
                return frames, voiced

    def hear(self, frames):
        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        t = time.time()
        with MLX_LOCK:
            text = transcribe(self.tr, audio)
        self.t_asr = time.time() - t
        return text

    def reply(self, text, t_end):
        """一句講完了：判斷要不要回、怎麼回。"""
        t0 = time.time()
        q = text
        if WEATHER_WORDS.search(text) and weather_home_cached():  # 問天氣：家裡的資料先備好，一次問完
            q = weather_home_prompt(text)
        while NOTICES:
            q = f"（系統通知：{NOTICES.pop(0)}）\n" + q
        tag, deltas = peek_tag(self.fast.ask(q))
        self.t_brain = time.time() - t0
        if tag == "WAIT":
            PANEL.add(text, "還沒講完")
            return "WAIT"
        if tag == "SKIP":
            PANEL.add(text, "略過（不是問我）")
            log("（聽到一段話，不是在問我，略過）")  # 不記內容，保護隱私
            return "SKIP"
        log(f"🦻 你說：{text}")
        hidden = []
        if isinstance(tag, tuple):  # 問天氣：先抓資料，再請快腦照資料回答
            deltas = self.fast.ask(weather_context(tag[1], text))
            tag = None
        if tag == "DEEP":
            ack = say_now("好，我想一下。")
            log(f"🧠 交給深腦（{DEEP_MODEL}）")
            ctx = "\n".join(f"她：{u}\n你：{a}" for u, a in self.history[-6:])
            q = (f"（最近的對話）\n{ctx}\n\n（她現在問）{text}") if ctx else text
            ans, first = speak_stream(split_tags(self.deep.ask(q), hidden), after=ack)
        else:
            # 11-13 她說：開頭不用先說「好」，反應速度跟直接回答差不多，直接回答就好
            ans, first = speak_stream(split_tags(deltas, hidden))
        log(f"🧠 賈維斯：{ans}")
        if OVERFLOW:
            ok = tg_send(ans)
            log(f"✂️ 太長只念前面，全文{'已傳 TG' if ok else '傳 TG 失敗'}")
            speak("後面比較長，我把全文傳到你的 TG 了。" if ok else "後面比較長，我先講到這裡。")
        if first:
            log(f"⏱️ 你講完到回答開口 {first - t_end + PAUSE_SEC:.1f}s"
                f"（等停頓 {PAUSE_SEC}／耳朵 {getattr(self, 't_asr', 0):.1f}／腦 {self.t_brain:.1f}"
                f"／嘴 {first - t0 - self.t_brain:.1f}）")
        tail = "".join(hidden)
        if "[[TG]]" in tail:
            msg = tail.split("[[TG]]", 1)[1].split("[[/TG]]", 1)[0].strip()
            ok = msg and tg_send(msg)
            log(f"📨 TG {'已傳' if ok else '失敗'}：{msg[:60]}")
            if not ok:
                speak("抱歉，TG 沒傳成功。")
        for name in re.findall(r"\[\[OPEN\]\](.*?)\[\[/OPEN\]\]", tail, re.S):
            if not open_app(name):
                speak("抱歉，打不開這個 App。")
        for query in re.findall(r"\[\[READ\]\](.*?)\[\[/READ\]\]", tail, re.S):
            self.read_report(query.strip())
        for spec in re.findall(r"\[\[REPORT\]\](.*?)\[\[/REPORT\]\]", tail, re.S):
            threading.Thread(target=write_report, args=(spec,), daemon=True).start()
        for name in re.findall(r"\[\[HOME\]\](.*?)\[\[/HOME\]\]", tail, re.S):
            if not run_home(name.strip(), self.home):
                speak("抱歉，這個家電指令沒有成功。")
        homes = re.findall(r"\[\[HOME\]\](.*?)\[\[/HOME\]\]", tail, re.S)
        PANEL.add(text, "家電" if homes else ("深想" if tag == "DEEP" else "回答"),
                  ans or ("執行：" + "、".join(h.strip() for h in homes)))
        self.history.append((text, ans))
        return "DONE"

    def read_report(self, query):
        local = REPORT_DIR.exists() and any(REPORT_DIR.glob("*.md"))
        if not local:
            say_now("好，我去 Notion 拿一下報告。")
        found = find_local(query)
        titles = report_titles()
        if not found and titles:  # 耳朵常聽錯（北海道→被海盜），請快腦從清單挑最像的
            pick = "".join(self.fast.ask(
                "（找報告）她要念的報告，關鍵字聽起來像「" + query + "」，語音辨識可能聽錯成諧音。"
                "現有報告：" + "、".join(titles) + "。只回最像的那個完整標題，都不像就只回 NONE，不要其他字。")).strip()
            pick = next((t for t in titles if t in pick or pick in t), "") if pick and "NONE" not in pick else ""
            log(f"📖 「{query}」→ 猜是「{pick or '都不像'}」")
            found = find_local(pick) if pick else None
        if not found:
            found = fetch_report(query)
        if not found:
            log(f"📖 找不到報告：{query}")
            speak("寶貝，我找不到那份報告耶，你再跟我說一下是哪一份？")
            return
        title, text = found
        log(f"📖 念報告：{title}")
        ans, _ = speak_stream(self.fast.ask(READ_PROMPT.format(title=title, text=text)), limit=1500)
        log(f"🧠 賈維斯（念報告）：{ans}")
        PANEL.add(f"念報告：{title}", "念報告", ans)
        self.history.append((f"念報告 {title}", ans))

    def run(self):
        speak("賈維斯已經上線，隨時聽你吩咐。")
        log("✅ 準備好了：一直在聽，直接問我就好。打字、雜音不會理。"
            "說「賈維斯暫停」我就先休息，「賈維斯關機」結束。")
        listening = True
        PANEL.serve()
        PANEL.mode = "一直聽"
        with self.mic.stream:
            # 開機自動啟動時，麥克風沒權限 macOS 不會報錯、只給全靜音 → 檢查一下，沒聲音就傳 TG 提醒
            level = (max(float(np.abs(self.mic.read(timeout=5)).max()) for _ in range(12))
                     if PANEL.mic_on else -1)  # 關閉收音中就不檢查
            log(f"🎤 麥克風檢查：最大音量 {level:.0f}" if level >= 0 else "🔇 目前是關閉收音（控制台可打開）")
            if level == 0:
                log("⚠️ 麥克風收不到任何聲音（可能沒權限）")
                tg_send("賈維斯開機了，但麥克風收不到聲音。請到「系統設定 → 隱私權與安全性 → 麥克風」"
                        "把 Python 打開，再跟我說一聲。")
            was_on = PANEL.mic_on
            while True:
                if PANEL.mic_on != was_on:
                    was_on = PANEL.mic_on
                    self.mic.drain()
                PANEL.mode = "一直聽" if listening else "暫停（等 Hey Jarvis）"
                if not PANEL.mic_on:
                    time.sleep(0.2)
                    continue
                if not listening:
                    f = self.mic.read()
                    scores = self.ww.predict(f)
                    if max(scores.values() or [0]) >= WAKE_THRESHOLD:
                        self.ww.reset()
                        listening = True
                        log("👂 Hey Jarvis → 恢復一直聽")
                        say_now("我在").wait()
                        self.mic.drain()
                    continue

                frames, voiced = self.listen()
                if voiced < MIN_VOICE_SEC:
                    continue  # 太短：咳嗽、碰撞聲
                while True:
                    t_end = time.time()
                    text = self.hear(frames)
                    if not text or looks_bogus(text) or looks_noise(text):
                        if text:
                            PANEL.add(text, "雜音" if looks_noise(text) else "聽不清")
                        result = "SKIP"
                        break
                    plain = text.replace(" ", "")
                    if re.search(r"[賈贾][維维]斯.{0,2}(關機|关机|下線|下线)", plain):
                        speak("好，賈維斯下線，再見。")
                        log("🛑 她說關機")
                        return
                    if re.search(r"[賈贾][維维]斯.{0,2}(暫停|暂停|休息|別聽|别听)", plain):
                        speak("好，我先休息。要找我說 Hey Jarvis。")
                        log("⏸️ 暫停一直聽，等 Hey Jarvis")
                        listening = False
                        self.ww.reset()
                        result = "PAUSE"
                        break
                    result = self.reply(text, t_end)
                    if result != "WAIT":
                        break
                    # 還沒講完：回「嗯～」，接著錄；她一直沒接下去就當作講完了
                    log(f"（還沒講完：{text}）→ 嗯～")
                    say_now("嗯～")
                    more, v2 = self.listen(frames, give_up=GIVEUP_SEC)
                    if len(more) == len(frames):
                        t_end = time.time()
                        result = self.reply(text + "\n（她停下來了，當作講完了）", t_end)
                        break
                    frames = more
                self.mic.drain()


def main():
    import fcntl
    global _lock
    _lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("賈維斯已經在跑了，不用再開一個。")
        return
    Jarvis().run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n賈維斯下線。再見 👋")
