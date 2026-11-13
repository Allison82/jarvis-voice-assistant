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
         "小紅書, 賈維斯, 玉山, 台灣, "
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
    "【領域】（可改成你自己的工作領域；這裡以 IC 驗證為例）主人常聊半導體／IC 驗證：這種情境下 RX 是接收端 receiver、TX 是傳送端 transmitter，不是醫療處方 Rx；"
    "SerDes、PCIe、DDR、UVM、SystemVerilog、testbench、DUT、RTL、waveform、coverage、assertion、eye diagram、jitter、BER、PHY、lane 這些都是硬體訊號／驗證的意思，往這個方向解釋。\n"
    "回答要適合直接念出來：不要用條列、emoji、括號註解、Markdown。"
    "第一句一定要很短（中文 10 個字、英文 8 個字以內），先講結論，再補充。"
    "中文句子裡的數字和單位寫成念得出來的中文（例如三千九百五十二公尺）；英文句子用英文念法。\n"
    "【傳 TG】你可以傳 Telegram 訊息給主人自己（透過使用者自己的 Telegram bot，只能傳給本人）。"
    "她說「傳到 TG」「傳給我」「記下來傳我」之類的時候，先用一句口語說你傳了什麼，"
    "然後在回覆最後另起一行輸出 [[TG]]要傳的完整內容[[/TG]]。要傳的內容可以比口語詳細、可以分行，這段不會被念出來。"
)

FAST_PROMPT = STYLE + (
    "\n【回覆開頭的判斷】你一直在聽主人房間裡的聲音。每次收到文字，先決定："
    "\n- 主人明顯還沒講完（句子斷在一半，例如「然後」「就是」「我想問那個」「幫我查一下」後面沒東西）→ 整個回覆只輸出 [[WAIT]]"
    "\n- 不是在跟你說話（自言自語、抒發心情、跟別人聊天、講電話、電視影片的聲音、無意義的片段；沒有問句也沒有請你做事）→ 整個回覆只輸出 [[SKIP]]"
    "\n- 需要大量思考（多步驟推理、規劃行程、比較分析、寫長內容、專業問題、計算）→ 整個回覆只輸出 [[DEEP]]"
    "\n- 其他一般問題、閒聊、簡單請求 → 直接回答，1 到 3 句話，不要加任何標記。"
    "\n收到「（她停下來了，當作講完了）」時，不可以再回 [[WAIT]]。"
)

DEEP_PROMPT = STYLE + "\n這是需要仔細想的問題。想清楚後用 3 到 6 句口語回答，條理清楚但不要列點。"

# ---- 載入 Nemotron（耳朵）----
from mlx_audio.stt.models.nemotron_transcribe.nemotron_transcribe import ASR_PROMPT
from nemo_mlx.pipeline import Transcriber


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ======================== 🎤 收音 ========================

class Mic:
    """麥克風聲音放進佇列；處理事情時聲音會先排隊，不會漏掉她搶話的部分。"""

    def __init__(self):
        self.q = queue.Queue()
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                                     blocksize=FRAME, callback=self._cb)

    def _cb(self, indata, frames, t, status):
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


def speak_stream(deltas, after=None):
    """邊想邊念：腦吐出一句就馬上產生聲音、一段一段排進播放佇列。
    after 不用特別等：「好」已經先排在佇列前面了。回傳 (整段文字, 第一聲時間)。"""
    said, first_sound = [], None
    for piece in chunks(deltas):
        said.append(piece)
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
        if not s.startswith("[") or len(s) >= 8:
            break
    s = head.lstrip()
    m = re.match(r"\[\[(WAIT|SKIP|DEEP)\]\]", s)
    if m:
        for _ in deltas:  # 把剩下的讀完，下一題才不會接到殘渣
            pass
        return m.group(1), None

    def rest():
        yield head
        yield from deltas
    return None, rest()


def tg_send(text):
    """用 Bot API 直接傳給使用者本人（讀 .env 的 bot token + access.json 的 chat_id）。
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
        self.fast = Brain(FAST_MODEL, FAST_PROMPT)
        self.deep = Brain(DEEP_MODEL, DEEP_PROMPT)
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
        with MLX_LOCK:
            return transcribe(self.tr, audio)

    def reply(self, text, t_end):
        """一句講完了：判斷要不要回、怎麼回。"""
        tag, deltas = peek_tag(self.fast.ask(text))
        if tag == "WAIT":
            return "WAIT"
        if tag == "SKIP":
            log("（聽到一段話，不是在問我，略過）")  # 不記內容，保護隱私
            return "SKIP"
        log(f"🦻 你說：{text}")
        hidden = []
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
        if first:
            log(f"⏱️ 你講完到回答開口 {first - t_end + PAUSE_SEC:.1f}s")
        tail = "".join(hidden)
        if "[[TG]]" in tail:
            msg = tail.split("[[TG]]", 1)[1].split("[[/TG]]", 1)[0].strip()
            ok = msg and tg_send(msg)
            log(f"📨 TG {'已傳' if ok else '失敗'}：{msg[:60]}")
            if not ok:
                speak("抱歉，TG 沒傳成功。")
        self.history.append((text, ans))
        return "DONE"

    def run(self):
        speak("賈維斯已經上線，隨時聽你吩咐。")
        log("✅ 準備好了：一直在聽，直接問我就好。打字、雜音不會理。"
            "說「賈維斯暫停」我就先休息，「賈維斯關機」結束。")
        listening = True
        with self.mic.stream:
            # 開機自動啟動時，麥克風沒權限 macOS 不會報錯、只給全靜音 → 檢查一下，沒聲音就傳 TG 提醒
            level = max(float(np.abs(self.mic.read(timeout=5)).max()) for _ in range(12))
            log(f"🎤 麥克風檢查：最大音量 {level:.0f}")
            if level == 0:
                log("⚠️ 麥克風收不到任何聲音（可能沒權限）")
                tg_send("賈維斯開機了，但麥克風收不到聲音。請到「系統設定 → 隱私權與安全性 → 麥克風」"
                        "把 Python 打開，再跟我說一聲。")
            while True:
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
                    if not text or looks_bogus(text):
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
