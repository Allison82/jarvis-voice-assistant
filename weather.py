"""賈維斯的天氣（2026-11-15）。

有 Windy 金鑰（~/jarvis/windy_key.txt）就用 Windy Point Forecast API；
沒有就用 Open-Meteo（免費、免金鑰、真實資料）。
注意：Windy 免費的 Testing 金鑰回傳的是「隨機打亂、稍微改過」的假資料，
要正式用得買 Professional（990 歐元／年）。windy_key.txt 第二行寫 testing
會提醒腦「這是測試資料」，避免把假數字當真念出來。

回傳一段給腦看的精簡文字（台灣時間，每 3 小時一筆＋每天高低溫／雨）。
"""
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

KEY_FILE = Path.home() / "jarvis" / "windy_key.txt"
HOME_LATLON = (25.03, 121.56)   # 預設：她家（改成你家的座標）
TZ = 8 * 3600                   # 台灣時間

WMO = {0: "晴", 1: "大致晴", 2: "多雲", 3: "陰", 45: "霧", 48: "霧", 51: "毛毛雨", 53: "毛毛雨",
       55: "毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨", 80: "陣雨", 81: "陣雨", 82: "大陣雨",
       95: "雷雨", 96: "雷雨", 99: "雷雨"}


def _windy_key():
    try:
        lines = KEY_FILE.read_text().split()
        return (lines[0], len(lines) > 1 and lines[1].lower() == "testing") if lines else (None, False)
    except OSError:
        return None, False


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=8))


def _get(url):
    return json.load(urllib.request.urlopen(url, timeout=8))


WEEK = "一二三四五六日"


def _hm(ts):
    t = time.gmtime(ts + TZ)
    return f"{t.tm_mon}/{t.tm_mday}(週{WEEK[t.tm_wday]}) {t.tm_hour}點"


def _wd(ymd):
    return "週" + WEEK[time.strptime(ymd, "%Y-%m-%d").tm_wday]


def _windy(lat, lon, key):
    d = _post("https://api.windy.com/api/point-forecast/v2", {
        "lat": round(lat, 2), "lon": round(lon, 2), "model": "gfs",
        "parameters": ["temp", "precip", "wind", "rh", "lclouds"],
        "levels": ["surface"], "key": key})
    rows, days = [], {}
    now = time.time()
    for i, ms in enumerate(d["ts"]):
        ts = ms / 1000
        if ts < now - 3600 or ts > now + 5 * 86400:
            continue
        temp = d["temp-surface"][i] - 273.15          # K → °C
        rain = d.get("past3hprecip-surface", [0] * len(d["ts"]))[i] or 0  # 公尺 → 下面換毫米
        rain_mm = rain * 1000
        u, v = d["wind_u-surface"][i], d["wind_v-surface"][i]
        wind = math.hypot(u, v)
        rh = d.get("rh-surface", [None] * len(d["ts"]))[i]
        day = time.strftime("%m/%d", time.gmtime(ts + TZ)) + "(" + _wd(time.strftime("%Y-%m-%d", time.gmtime(ts + TZ))) + ")"
        lo, hi, rr = days.get(day, (99, -99, 0))
        days[day] = (min(lo, temp), max(hi, temp), rr + rain_mm)
        if ts < now + 86400:
            rows.append(f"{_hm(ts)} {temp:.0f}度 前3小時雨量{rain_mm:.1f}毫米 風{wind:.0f}m/s"
                        + (f" 濕度{rh:.0f}%" if rh is not None else ""))
    daily = [f"{k} 低{lo:.0f}度 高{hi:.0f}度 總雨量{rr:.0f}毫米" for k, (lo, hi, rr) in days.items()]
    return "Windy（GFS 模型）", rows, daily


def _open_meteo(lat, lon):
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "Asia/Taipei", "forecast_days": 5,
        "hourly": "temperature_2m,precipitation_probability,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
        "wind_speed_unit": "ms"})
    d = _get("https://api.open-meteo.com/v1/forecast?" + q)
    h, rows = d["hourly"], []
    now = time.strftime("%Y-%m-%dT%H", time.gmtime(time.time() + TZ))
    start = next((i for i, t in enumerate(h["time"]) if t[:13] >= now), 0)
    for i in range(start, min(start + 24, len(h["time"])), 3):
        t = h["time"][i]
        rows.append(f"{int(t[5:7])}/{int(t[8:10])} {int(t[11:13])}點 {h['temperature_2m'][i]:.0f}度 "
                    f"{WMO.get(h['weather_code'][i], '')} 降雨機率{h['precipitation_probability'][i]}% "
                    f"風{h['wind_speed_10m'][i]:.0f}m/s")
    dd = d["daily"]
    daily = [f"{t[5:]}({_wd(t)}) {WMO.get(c, '')} 低{lo:.0f}度 高{hi:.0f}度 降雨機率{p}%"
             for t, c, lo, hi, p in zip(dd["time"], dd["weather_code"], dd["temperature_2m_min"],
                                        dd["temperature_2m_max"], dd["precipitation_probability_max"])]
    return "Open-Meteo", rows, daily


def forecast(lat=None, lon=None, place="她家（台北）"):
    """回傳 (給腦看的文字, 用哪個來源)。失敗回傳 (None, 原因)。"""
    if lat is None:
        lat, lon = HOME_LATLON
    key, testing = _windy_key()
    try:
        src, rows, daily = _windy(lat, lon, key) if key else _open_meteo(lat, lon)
    except Exception as e:
        if not key:
            return None, str(e)
        try:  # Windy 掛了就退回 Open-Meteo
            src, rows, daily = _open_meteo(lat, lon)
            testing = False
        except Exception as e2:
            return None, f"{e} / {e2}"
    note = "（注意：這是 Windy 測試金鑰的打亂資料，數字不準，回答時要提醒她）" if key and testing and src.startswith("Windy") else ""
    text = (f"（天氣資料，來源 {src}，地點 {place}，台灣時間，現在是 {_hm(time.time())}）{note}\n"
            "未來 24 小時：\n" + "\n".join(rows) + "\n未來幾天：\n" + "\n".join(daily))
    return text, src


if __name__ == "__main__":
    print(forecast()[0])
