"""埋伏股選股器主流程。

階段一：全市場日K（yfinance）→ 技術面粗篩（打底 / 窒息量 / 均線糾結）
階段二：入圍者逐檔抓 FinMind（融資、法人、月營收、PER/PBR、股權分散）
階段三：前段班查媒體聲量（Google News、PTT），選用分點資料
輸出：docs/results.json + docs/results.md（GitHub Pages 讀取）
"""
import argparse
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from . import official, prices, report, sentiment
from . import signals as S
from .finmind import FinMind, QuotaExceeded
from .net import log
from .universe import load_universe

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(p):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def combine(sigs: dict, weights: dict):
    num = den = 0.0
    total = sum(weights.values())
    for k, w in weights.items():
        s = sigs.get(k, {}).get("score")
        if s is None:
            continue
        num += w * s
        den += w
    return (100 * num / den if den else 0.0), (den / total if total else 0.0)


# ---------------------------------------------------------------- 階段一
def stage1(universe, px_map, cfg):
    tc, uc = cfg["technical"], cfg["universe"]
    rows = []
    for r in universe.itertuples():
        px = px_map.get(r.code)
        if px is None or len(px) < tc["base"]["min_days"]:
            continue
        if px["close"].iloc[-1] < uc["min_price"]:
            continue
        if px["volume"].iloc[-20:].mean() / 1000 < uc["min_avg_lots"]:
            continue
        sig = {"base": S.sig_base(px, tc["base"]),
               "volume_dry": S.sig_volume_dry(px, tc["volume"]),
               "ma_squeeze": S.sig_ma_squeeze(px, tc["ma"])}
        spread = sig["ma_squeeze"]["metrics"].get("spread")
        if spread is None or spread > cfg["stage1"]["max_spread"]:
            continue
        if cfg["stage1"]["require_base"] and not (sig["base"]["score"] or 0) > 0:
            continue
        w = {k: cfg["weights"][k] for k in sig}
        tech, _ = combine(sig, w)
        rows.append({"code": r.code, "name": r.name, "market": r.market,
                     "industry": r.industry, "signals": sig, "tech": tech,
                     "trigger": S.sig_trigger(px, tc["trigger"])})
    rows.sort(key=lambda x: -x["tech"])
    return rows


# ---------------------------------------------------------------- 階段二
def holding_from_finmind(df):
    if df is None or df.empty:
        return None
    lvl = df["HoldingSharesLevel"].astype(str)
    big = df[lvl.str.contains("1,000,001")].groupby("date")["percent"].sum()
    tot = df[lvl.str.lower() == "total"].groupby("date")["people"].sum()
    h = pd.DataFrame({"big_pct": big, "holders": tot}).dropna().reset_index()
    h["date"] = pd.to_datetime(h["date"])
    return h if len(h) else None


def enrich(row, fm, px, cfg, tdcc_hist, insider_hist, themes):
    code, today = row["code"], date.today()
    ago = lambda d: (today - timedelta(days=d)).isoformat()  # noqa: E731
    sig, cc = row["signals"], cfg["chips"]

    # 融資
    m = fm.get("TaiwanStockMarginPurchaseShortSale", code, ago(220))
    bal = (m.set_index("date")["MarginPurchaseTodayBalance"].astype(float).sort_index()
           if not m.empty else None)
    sig["margin"] = S.sig_margin(bal, cc["margin"])

    # 法人（主力替代指標）
    ins = fm.get("TaiwanStockInstitutionalInvestorsBuySell", code, ago(45))
    net = None
    if not ins.empty:
        ins["net"] = ins["buy"].astype(float) - ins["sell"].astype(float)
        net = ins.groupby("date")["net"].sum().sort_index()
    row["_inst"] = net
    sig["smart_money"] = S.sig_smart_money(net, px, cc["smart_money"])

    # 月營收 & 估值
    rev = S.monthly_revenue(fm.get("TaiwanStockMonthRevenue", code, ago(40 * 31)))
    per = fm.get("TaiwanStockPER", code, ago(5 * 366))
    per_latest = float(per.sort_values("date")["PER"].iloc[-1]) if not per.empty else None
    sig["turnaround"] = S.sig_turnaround(rev, cfg["fundamental"], per_latest)
    sig["yoy_turn"] = S.sig_yoy_turn(rev, cfg["fundamental"])
    sig["valuation"] = S.sig_valuation(per, px, cfg["valuation"]["price_position_years"])
    sig["theme"] = S.sig_theme(code, row["industry"], themes)

    # 千張大戶：FinMind 優先，退回自行累積的集保快照
    hold = None
    if cfg["finmind"]["use_holding_shares"]:
        hold = holding_from_finmind(fm.get("TaiwanStockHoldingSharesPer", code, ago(70)))
    if hold is None and not tdcc_hist.empty:
        hold = tdcc_hist[tdcc_hist["code"] == code][["date", "big_pct", "holders"]]
    sig["big_holder"] = S.sig_big_holder(hold, cc["big_holder"])

    # 董監持股（每月累積快照）
    ih = insider_hist[insider_hist["code"] == code] if not insider_hist.empty else None
    sig["insider"] = S.sig_insider(ih, cc["insider"])


def branch_signal(fm, code, px, cfg):
    days = [d.strftime("%Y-%m-%d") for d in px.index[-cfg["finmind"]["branch_days"]:]]
    frames = [fm.get("TaiwanStockTradingDailyReport", code, d, d) for d in days]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return None, ""
    df = pd.concat(frames)
    df["net"] = df["buy"].astype(float) - df["sell"].astype(float)
    by = df.groupby("securities_trader")["net"].sum().sort_values(ascending=False)
    top_n = cfg["chips"]["smart_money"]["branch_top"]
    vol = px["volume"].iloc[-len(days):].sum()
    ratio = by.head(top_n).sum() / vol if vol else 0
    top = "、".join(f"{k}({v / 1000:+.0f}張)" for k, v in by.head(3).items())
    return ratio, f"前{top_n}大買超分點占量 {ratio:.1%}：{top}"


# ---------------------------------------------------------------- 主程式
def run(limit=0):
    cfg = load_yaml(ROOT / "config.yaml")
    themes = (load_yaml(ROOT / "themes.yaml") or {}).get("themes", {})
    data_dir, docs_dir = ROOT / cfg["paths"]["data"], ROOT / cfg["paths"]["docs"]
    token = os.getenv("FINMIND_TOKEN")
    fc = cfg["finmind"]
    fm = FinMind(token, fc["min_interval_with_token" if token else "min_interval_without_token"])
    warnings = []
    if not token:
        warnings.append("未設定 FINMIND_TOKEN，深入分析檔數降為 "
                        f"{cfg['stage2']['max_without_token']} 檔")

    uni = load_universe(fm)
    limit = limit or cfg["universe"]["limit"]
    if limit:
        uni = uni.head(limit)

    for fn, name in ((official.update_tdcc, "集保股權分散"), (official.update_insider, "董監持股")):
        try:
            fn(data_dir)
        except Exception as e:
            warnings.append(f"{name}快照更新失敗：{e}")
            log.warning("%s 更新失敗：%s", name, e)
    tdcc_hist = official.load_tdcc(data_dir)
    insider_hist = official.load_insider(data_dir)

    px_map = prices.download_prices(uni, cfg["prices"]["period"], cfg["prices"]["chunk"])
    if len(px_map) < 0.5 * len(uni):
        warnings.append(f"價格僅取得 {len(px_map)}/{len(uni)} 檔，yfinance 可能被限流")

    s1 = stage1(uni, px_map, cfg)
    log.info("階段一入圍 %d 檔", len(s1))
    cap = cfg["stage2"]["max_with_token" if token else "max_without_token"]
    finalists = s1[:cap]

    done = []
    for i, row in enumerate(finalists, 1):
        try:
            enrich(row, fm, px_map[row["code"]], cfg, tdcc_hist, insider_hist, themes)
            done.append(row)
        except QuotaExceeded as e:
            warnings.append(f"FinMind 額度用盡，只完成 {len(done)} 檔深入分析：{e}")
            break
        except Exception as e:
            log.exception("分析 %s 失敗", row["code"])
            warnings.append(f"{row['code']} 分析失敗：{e}")
        if i % 10 == 0:
            log.info("深入分析 %d/%d（FinMind 呼叫 %d 次）", i, len(finalists), fm.calls)

    for row in done:
        row["score"], row["coverage"] = combine(row["signals"], cfg["weights"])
    done.sort(key=lambda r: -r["score"])

    # 媒體聲量：只查前段班
    sc = cfg["sentiment"]
    for row in done[:sc["top_n"]]:
        n, p = sentiment.lookup(row["name"], row["code"], sc["days"])
        row["signals"]["media"] = S.sig_media(n, p, sc)
    # 分點（選用）
    if fc["use_branch_report"]:
        try:
            for row in done[:fc["branch_top_n"]]:
                ratio, txt = branch_signal(fm, row["code"], px_map[row["code"]], cfg)
                if ratio is not None:
                    row["signals"]["smart_money"] = S.sig_smart_money(
                        row.get("_inst"), px_map[row["code"]], cfg["chips"]["smart_money"],
                        ratio, txt)
        except QuotaExceeded as e:
            warnings.append(f"分點資料額度用盡：{e}")

    items = []
    for row in done:
        row["score"], row["coverage"] = combine(row["signals"], cfg["weights"])
        if row["coverage"] < cfg["output"]["min_coverage"]:
            continue
        sg = row["signals"]
        flags = []
        if row["trigger"]["fired"]:
            flags.append("已點火")
        if (sg["theme"]["score"] or 0) >= 0.4 and (sg.get("media", {}).get("score") or 0) >= 0.7:
            flags.append("題材未被發現")
        if (sg["yoy_turn"]["metrics"].get("turned")):
            flags.append("營收轉正")
        px = px_map[row["code"]]
        items.append({
            "code": row["code"], "name": row["name"], "market": row["market"],
            "industry": row["industry"], "score": round(row["score"], 1),
            "coverage": round(row["coverage"], 2), "close": round(float(px["close"].iloc[-1]), 2),
            "date": px.index[-1].strftime("%Y-%m-%d"), "trigger": row["trigger"],
            "flags": flags, "signals": sg, "links": sentiment.links(row["name"], row["code"]),
        })
    items.sort(key=lambda x: -x["score"])
    items = items[:cfg["output"]["top_n"]]

    n_weeks = tdcc_hist["date"].nunique() if not tdcc_hist.empty else 0
    if n_weeks < cfg["chips"]["big_holder"]["min_weeks"]:
        warnings.append(f"自行累積的集保快照僅 {n_weeks} 週（FinMind 股權分散表不可用時才會用到）")
    meta = {"generated_at": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
            "universe": len(uni), "stage1": len(s1), "stage2": len(done),
            "finmind_calls": fm.calls, "weights": cfg["weights"], "warnings": warnings}
    report.write(items, meta, docs_dir)
    log.info("完成：上榜 %d 檔", len(items))
    return items, meta


def cli():
    ap = argparse.ArgumentParser(description="台股埋伏股選股器")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 檔（除錯用）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.limit)


if __name__ == "__main__":
    cli()
