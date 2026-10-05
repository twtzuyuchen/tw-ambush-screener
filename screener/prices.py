"""全市場日 K：用 yfinance 批次下載（還原權息，避免除息缺口破壞箱型判斷）。"""
import time

import pandas as pd

from .net import log

SUFFIX = {"twse": ".TW", "tpex": ".TWO"}


def _split(data: pd.DataFrame, tickers):
    out = {}
    if data is None or data.empty:
        return out
    multi = isinstance(data.columns, pd.MultiIndex)
    for t in tickers:
        try:
            df = data[t] if multi else data
        except KeyError:
            continue
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df = df.dropna(subset=["close"])
        if len(df):
            df.index = pd.to_datetime(df.index).tz_localize(None)
            out[t] = df
    return out


def download_prices(universe: pd.DataFrame, period="3y", chunk=150) -> dict:
    import yfinance as yf

    tick2code = {f"{r.code}{SUFFIX[r.market]}": r.code for r in universe.itertuples()}
    tickers = list(tick2code)
    result = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        for attempt in range(3):
            try:
                data = yf.download(batch, period=period, interval="1d",
                                   group_by="ticker", auto_adjust=True,
                                   threads=True, progress=False)
                got = _split(data, batch)
                break
            except Exception as e:
                log.warning("yfinance 批次 %d 第 %d 次失敗：%s", i, attempt + 1, e)
                got = {}
                time.sleep(10 * (attempt + 1))
        for t, df in got.items():
            result[tick2code[t]] = df
        log.info("價格下載 %d/%d，累計 %d 檔", min(i + chunk, len(tickers)),
                 len(tickers), len(result))
        time.sleep(2)
    return result
