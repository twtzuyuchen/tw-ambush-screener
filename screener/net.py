"""共用 HTTP 工具：重試、User-Agent、數字清洗。"""
import logging
import re

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("screener")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": UA, "Accept-Language": "zh-TW,zh;q=0.9"})
    return s


SESSION = make_session()


def get_json(url, params=None, timeout=40, headers=None):
    r = SESSION.get(url, params=params, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()


def get_text(url, params=None, timeout=40, encoding=None):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    if encoding:
        r.encoding = encoding
    return r.text


def to_num(x):
    """'1,234' / '--' / '' / None → float。"""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = re.sub(r"[,\s%]", "", str(x))
    if s in ("", "-", "--", "---", "N/A", "nan"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def find_col(df: pd.DataFrame, *candidates):
    """依序找出第一個欄名包含候選字串的欄位（官方 API 欄名偶有變動）。"""
    cols = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    for cand in candidates:
        for orig, c in zip(df.columns, cols):
            if cand in c:
                return orig
    return None


def is_common_stock(code: str) -> bool:
    return bool(re.fullmatch(r"[1-9]\d{3}", str(code)))
