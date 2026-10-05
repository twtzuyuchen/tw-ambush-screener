"""股票池：上市＋上櫃普通股（排除 ETF、權證、DR 以外的非四碼證券）。"""
import pandas as pd

from .net import find_col, get_json, is_common_stock, log

TWSE_DAY_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_DAY_ALL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"


def _from_finmind(fm) -> pd.DataFrame:
    df = fm.get("TaiwanStockInfo")
    if df.empty:
        return df
    df = df[df["type"].isin(["twse", "tpex"])]
    df = df[df["stock_id"].map(is_common_stock)]
    df = df[~df["industry_category"].isin(["ETF", "ETN", "Index", "大盤", "存託憑證"])]
    agg = (df.groupby("stock_id")
             .agg(name=("stock_name", "first"), market=("type", "first"),
                  industry=("industry_category",
                            lambda s: "、".join(sorted(set(map(str, s))))))
             .reset_index().rename(columns={"stock_id": "code"}))
    return agg


def _from_openapi() -> pd.DataFrame:
    frames = []
    for market, url in (("twse", TWSE_DAY_ALL), ("tpex", TPEX_DAY_ALL)):
        try:
            df = pd.DataFrame(get_json(url))
            c = find_col(df, "SecuritiesCompanyCode", "Code", "代號")
            n = find_col(df, "CompanyName", "Name", "名稱")
            frames.append(pd.DataFrame({"code": df[c].astype(str).str.strip(),
                                        "name": df[n].astype(str).str.strip(),
                                        "market": market, "industry": ""}))
        except Exception as e:
            log.warning("openapi 股票清單 %s 失敗：%s", market, e)
    if not frames:
        return pd.DataFrame(columns=["code", "name", "market", "industry"])
    df = pd.concat(frames)
    return df[df["code"].map(is_common_stock)].drop_duplicates("code")


def load_universe(fm) -> pd.DataFrame:
    """FinMind 清單含產業別但也含已下市股票，用官方當日交易清單過濾。"""
    df = _from_finmind(fm)
    live = _from_openapi()
    if df.empty:
        log.warning("FinMind 股票清單取得失敗，改用證交所/櫃買 OpenAPI")
        df = live
    elif not live.empty:
        before = len(df)
        df = df[df["code"].isin(set(live["code"]))]
        log.info("排除已下市或停止交易：%d 檔", before - len(df))
    log.info("股票池：%d 檔", len(df))
    return df.sort_values("code").reset_index(drop=True)
