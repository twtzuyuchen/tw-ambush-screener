"""媒體與論壇聲量：Google News RSS + PTT 股板搜尋。

Dcard、Mobile01 有反爬蟲防護，未納入；請在結果頁手動點連結確認。
"""
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from urllib.parse import quote

from .net import get_text, log

GNEWS = "https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
PTT = "https://www.ptt.cc/bbs/Stock/search?q={q}"


def news_count(name: str, code: str, days: int):
    q = quote(f'"{name}" {code} when:{days}d')
    try:
        root = ET.fromstring(get_text(GNEWS.format(q=q)))
        return len(root.findall(".//item"))
    except Exception as e:
        log.warning("Google News %s 失敗：%s", code, e)
        return None


def ptt_count(name: str, days: int):
    try:
        html = get_text(PTT.format(q=quote(name)))
    except Exception as e:
        log.warning("PTT %s 失敗：%s", name, e)
        return None
    today = date.today()
    cutoff = today - timedelta(days=days)
    n = 0
    for m, d in re.findall(r'<div class="date">\s*(\d{1,2})/(\d{1,2})\s*</div>', html):
        try:
            dt = date(today.year, int(m), int(d))
        except ValueError:
            continue
        if dt > today:  # 跨年
            dt = dt.replace(year=today.year - 1)
        n += dt >= cutoff
    return n  # 第一頁最多 20 篇，足以判斷「乏人問津」


def links(name: str, code: str) -> dict:
    return {
        "news": f"https://news.google.com/search?q={quote(name + ' ' + code)}&hl=zh-TW",
        "ptt": PTT.format(q=quote(name)),
        "dcard": f"https://www.dcard.tw/search?query={quote(name)}&forum=stock",
        "mobile01": f"https://www.mobile01.com/googlesearch.php?q={quote(name)}",
        "goodinfo": f"https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={code}",
    }


def lookup(name: str, code: str, days: int):
    n = news_count(name, code, days)
    time.sleep(1.5)
    p = ptt_count(name, days)
    time.sleep(1.5)
    return n, p
