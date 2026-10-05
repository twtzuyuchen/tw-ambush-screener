"""官方開放資料的「累積快照」：每次執行存一份，讓 repo 本身變成歷史資料庫。

- 集保戶股權分散表（每週）：opendata 只提供最新一週，所以要自己累積。
- 董監事持股餘額（每月）：用來偵測內部人持股增加。
"""
import io
from pathlib import Path

import pandas as pd

from .net import find_col, get_json, get_text, is_common_stock, log, to_num

TDCC_URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
INSIDER_URLS = {
    "twse": "https://openapi.twse.com.tw/v1/opendata/t187ap11_L",
    "tpex": "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap11_O",
}


# ---------------- 集保股權分散 ----------------
def update_tdcc(data_dir: Path):
    text = get_text(TDCC_URL, encoding="utf-8")
    df = pd.read_csv(io.StringIO(text.lstrip("\ufeff")), dtype=str)
    c_date = find_col(df, "日期")
    c_code = find_col(df, "代號")
    c_lvl = find_col(df, "分級")
    c_ppl = find_col(df, "人數")
    c_pct = find_col(df, "比例")
    df[c_lvl] = df[c_lvl].str.strip()
    df[c_code] = df[c_code].str.strip()
    df = df[df[c_code].map(is_common_stock)]
    date = str(df[c_date].iloc[0]).strip()
    big = df[df[c_lvl] == "15"].set_index(c_code)[c_pct].map(to_num)   # 1,000,001 股以上
    tot = df[df[c_lvl] == "17"].set_index(c_code)[c_ppl].map(to_num)   # 合計人數
    snap = pd.DataFrame({"big_pct": big, "holders": tot}).dropna(how="all")
    snap.index.name = "code"
    out = data_dir / "tdcc" / f"{date}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        log.info("集保 %s 已存在，略過", date)
        return
    snap.to_csv(out)
    log.info("集保快照 %s：%d 檔", date, len(snap))


def load_tdcc(data_dir: Path) -> pd.DataFrame:
    files = sorted((data_dir / "tdcc").glob("*.csv.gz"))
    frames = []
    for f in files:
        d = pd.read_csv(f, dtype={"code": str})
        d["date"] = pd.to_datetime(f.name.split(".")[0], format="%Y%m%d")
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["code", "big_pct", "holders", "date"])
    return pd.concat(frames, ignore_index=True)


# ---------------- 董監事持股 ----------------
def _norm_ym(v) -> int | None:
    n = to_num(v)
    if pd.isna(n):
        return None
    n = int(n)
    if n < 200000:  # 民國年月，例如 11509
        n = (n // 100 + 1911) * 100 + n % 100
    return n


def update_insider(data_dir: Path):
    frames = []
    for market, url in INSIDER_URLS.items():
        try:
            df = pd.DataFrame(get_json(url))
            c_code = find_col(df, "公司代號", "SecuritiesCompanyCode", "Code")
            c_hold = find_col(df, "目前持股", "CurrentShareholding")
            c_ym = find_col(df, "資料年月", "YearMonth")
            if not (c_code and c_hold):
                log.warning("董監持股 %s 欄位無法辨識：%s", market, list(df.columns)[:12])
                continue
            df["_code"] = df[c_code].astype(str).str.strip()
            df["_hold"] = df[c_hold].map(to_num)
            df["_ym"] = df[c_ym].map(_norm_ym) if c_ym else None
            frames.append(df[["_code", "_hold", "_ym"]])
        except Exception as e:
            log.warning("董監持股 %s 失敗：%s", market, e)
    if not frames:
        return
    df = pd.concat(frames)
    df = df[df["_code"].map(is_common_stock)]
    ym = df["_ym"].dropna().mode()
    ym = int(ym.iloc[0]) if len(ym) else int(pd.Timestamp.today().strftime("%Y%m"))
    agg = df.groupby("_code")["_hold"].sum().rename("insider_shares")
    agg.index.name = "code"
    out = data_dir / "insider" / f"{ym}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out)  # 同月份覆寫：月中可能有補申報
    log.info("董監持股快照 %s：%d 檔", ym, len(agg))


def load_insider(data_dir: Path) -> pd.DataFrame:
    files = sorted((data_dir / "insider").glob("*.csv.gz"))
    frames = []
    for f in files:
        d = pd.read_csv(f, dtype={"code": str})
        d["ym"] = int(f.name.split(".")[0])
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["code", "insider_shares", "ym"])
    return pd.concat(frames, ignore_index=True)
