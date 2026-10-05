"""四大面向的訊號計算。

每個訊號回傳 {"score": 0~1 或 None, "detail": 中文說明, "metrics": {...}}；
score=None 代表資料不足，該訊號的權重不列入分母（見 main.combine）。
"""
import numpy as np
import pandas as pd


def res(score, detail, **metrics):
    s = None if score is None else float(np.clip(score, 0, 1))
    def clean(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.number)):
            return round(float(v), 4)
        return v
    m = {k: clean(v) for k, v in metrics.items()}
    return {"score": s, "detail": detail, "metrics": m}


def na(reason):
    return res(None, reason)


def lin(x, full, zero):
    """x 在 full 時得 1、在 zero 時得 0，中間線性。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(np.clip((x - zero) / (full - zero), 0, 1))


# =====================================================================
# 一、技術面：極致壓縮
# =====================================================================
def trend_r2(seg: pd.Series) -> float:
    y = np.log(seg.values.astype(float))
    x = np.arange(len(y))
    if np.std(y) == 0:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r * r)


def sig_base(px: pd.DataFrame, cfg) -> dict:
    """長期橫盤打底：找出最長的「振幅 ≤ max_range 且沒有明顯趨勢」視窗（120~250 日）。

    趨勢判斷用線性迴歸 R²：箱型/W底/圓底的 R² 低，緩漲緩跌的 R² 高。
    """
    c = px["close"]
    lo_d, hi_d, max_rng = cfg["min_days"], cfg["max_days"], cfg["max_range"]
    if len(c) < lo_d:
        return na("價格資料不足半年")
    found = None
    for w in range(min(hi_d, len(c)), lo_d - 1, -10):
        seg = c.iloc[-w:]
        rng = seg.max() / seg.min() - 1
        if rng <= max_rng and trend_r2(seg) <= cfg["max_trend_r2"]:
            found = (w, rng, seg.max(), seg.min())
            break
    if not found:
        seg = c.iloc[-lo_d:]
        rng = seg.max() / seg.min() - 1
        r2 = trend_r2(seg)
        why = f"振幅 {rng:.0%}" if rng > max_rng else f"呈單邊趨勢（R² {r2:.2f}）"
        return res(0, f"近{lo_d}日{why}，未形成打底結構", range=rng, r2=r2)
    w, rng, hi, lo = found
    pos = (c.iloc[-1] - lo) / (hi - lo) if hi > lo else 0.5
    pre = c.iloc[:-w]
    prior_drop = pre.max() / hi - 1 if len(pre) > 20 else None
    score = 0.5 + 0.3 * (w - lo_d) / max(hi_d - lo_d, 1) + 0.2 * (1 - rng / max_rng)
    detail = (f"約 {w // 5} 週箱型整理，區間 {lo:.2f}–{hi:.2f}（振幅 {rng:.0%}），"
              f"現價位於箱內 {pos:.0%}")
    if prior_drop is not None and prior_drop > 0.3:
        detail += f"；整理前高點比箱頂高 {prior_drop:.0%}，屬跌深後打底"
    return res(score, detail, days=w, range=rng, box_high=hi, box_low=lo,
               pos=pos, prior_drop=prior_drop)


def sig_volume_dry(px: pd.DataFrame, cfg) -> dict:
    """窒息量：20 日均量 / 年均量，以及近 60 日是否出現一年最低的 5 日均量。"""
    v = px["volume"].astype(float)
    n = min(cfg["long_days"], len(v))
    if n < 120:
        return na("成交量資料不足")
    v20 = v.iloc[-20:].mean()
    vlong = v.iloc[-n:].mean()
    if vlong <= 0:
        return na("無成交量")
    ratio = v20 / vlong
    r5 = v.rolling(5).mean()
    choke = bool(r5.iloc[-60:].min() <= r5.iloc[-n:].quantile(cfg["choke_quantile"]))
    score = 0.7 * lin(ratio, cfg["dry_ratio_full"], cfg["dry_ratio_zero"]) + 0.3 * choke
    detail = f"20日均量為年均量的 {ratio:.0%}"
    detail += "；近60日出現一年最低量（窒息量）" if choke else ""
    return res(score, detail, ratio=ratio, choke=choke, avg20_lots=v20 / 1000)


def ma_spread_series(c: pd.Series, periods) -> tuple[pd.DataFrame, pd.Series]:
    mas = pd.DataFrame({p: c.rolling(p).mean() for p in periods})
    spread = (mas.max(axis=1) - mas.min(axis=1)) / c
    return mas, spread


def sig_ma_squeeze(px: pd.DataFrame, cfg) -> dict:
    """均線糾結＋多頭排列。"""
    periods = cfg["periods"]
    c = px["close"]
    if len(c) < max(periods) + 20:
        return na("均線資料不足")
    mas, spread = ma_spread_series(c, periods)
    last = mas.iloc[-1]
    sp = float(spread.iloc[-1])
    max_sp = cfg["max_spread"]
    squeeze = lin(sp, 0, max_sp) if sp <= max_sp else 0.0
    vals = [last[p] for p in sorted(periods)]
    full_bull = all(a >= b for a, b in zip(vals, vals[1:]))
    long_p = max(periods)
    mid_p = sorted(periods)[-2]
    part_bull = last[mid_p] >= last[long_p] and c.iloc[-1] >= last[long_p]
    slope = mas[long_p].iloc[-1] / mas[long_p].iloc[-11] - 1
    days_tight = int((spread.iloc[-20:] <= max_sp).sum())
    order = 1.0 if full_bull else (0.5 if part_bull else 0.0)
    score = 0.55 * squeeze + 0.30 * order + 0.15 * (slope >= 0)
    label = "完全多頭排列" if full_bull else ("偏多（中期>長期）" if part_bull else "未成多頭排列")
    detail = (f"{'/'.join(map(str, periods))}日均線差距 {sp:.1%}，近20日有 {days_tight} 日"
              f"差距 ≤ {max_sp:.0%}；{label}；{long_p}日線{'上揚' if slope >= 0 else '下彎'}")
    return res(score, detail, spread=sp, full_bull=full_bull, days_tight=days_tight,
               ma_long_slope=slope)


def sig_trigger(px: pd.DataFrame, cfg) -> dict:
    """點火：帶量紅K突破箱頂。只標記，不計分。"""
    if len(px) < cfg["lookback"] + 21:
        return {"fired": False, "detail": ""}
    t = px.iloc[-1]
    prev_high = px["high"].iloc[-cfg["lookback"] - 1:-1].max()
    v20 = px["volume"].iloc[-21:-1].mean()
    gain = t["close"] / px["close"].iloc[-2] - 1
    fired = bool(t["close"] > prev_high and t["close"] > t["open"]
                 and t["volume"] >= cfg["vol_mult"] * v20 and gain >= cfg["min_gain"])
    detail = (f"今日帶量紅K突破{cfg['lookback']}日高點（漲 {gain:.1%}，量為均量 "
              f"{t['volume'] / v20:.1f} 倍）") if fired else ""
    return {"fired": fired, "detail": detail}


def price_position(px: pd.DataFrame, years: int) -> float:
    c = px["close"].iloc[-250 * years:]
    lo, hi = c.min(), c.max()
    return float((c.iloc[-1] - lo) / (hi - lo)) if hi > lo else 0.5


# =====================================================================
# 二、籌碼面
# =====================================================================
def sig_big_holder(hold: pd.DataFrame, cfg) -> dict:
    """hold: columns date, big_pct(千張大戶%), holders(總股東人數)，依日期排序。"""
    if hold is None or len(hold) < cfg["min_weeks"]:
        n = 0 if hold is None else len(hold)
        return na(f"股權分散資料僅 {n} 週（需 ≥{cfg['min_weeks']} 週）")
    seg = hold.sort_values("date").tail(cfg["weeks"])
    d_pct = seg["big_pct"].iloc[-1] - seg["big_pct"].iloc[0]
    up_ratio = float((seg["big_pct"].diff().dropna() > 0).mean())
    d_hold = seg["holders"].iloc[-1] / seg["holders"].iloc[0] - 1
    p1 = lin(d_pct, cfg["target_increase_pp"], 0)
    p2 = up_ratio if d_pct > 0 else 0
    p3 = lin(-d_hold, cfg["target_holder_drop"], 0)
    score = 0.45 * p1 + 0.25 * p2 + 0.30 * p3
    detail = (f"近 {len(seg)} 週千張大戶 {seg['big_pct'].iloc[0]:.2f}% → "
              f"{seg['big_pct'].iloc[-1]:.2f}%（{d_pct:+.2f} 個百分點，{up_ratio:.0%} 週上升）；"
              f"股東人數 {d_hold:+.1%}")
    return res(score, detail, d_pct=d_pct, up_ratio=up_ratio, d_holders=d_hold)


def sig_margin(bal: pd.Series, cfg) -> dict:
    """bal: 融資餘額（張），依日期排序。"""
    if bal is None or len(bal) < 60:
        return na("融資資料不足")
    b = bal.iloc[-cfg["lookback"]:].astype(float)
    if b.max() < cfg["min_max_balance"]:
        return na("融資餘額過小，不具參考性")
    cur, lo, hi = b.iloc[-1], b.min(), b.max()
    near_low = cur <= lo * (1 + cfg["near_low_tol"])
    chg60 = cur / b.iloc[-60] - 1 if b.iloc[-60] > 0 else 0
    p1 = 1.0 if near_low else 0.6 * (1 - (cur - lo) / (hi - lo)) if hi > lo else 0
    p2 = lin(-chg60, cfg["target_drop_60d"], 0)
    score = 0.5 * p1 + 0.5 * p2
    detail = (f"融資餘額 {cur:,.0f} 張，60日 {chg60:+.0%}"
              + (f"；處於近{len(b)}日低點" if near_low else ""))
    return res(score, detail, balance=cur, chg60=chg60, near_low=bool(near_low))


def cmf(px: pd.DataFrame, days: int) -> float:
    p = px.iloc[-days:]
    rng = (p["high"] - p["low"]).replace(0, np.nan)
    mfm = ((p["close"] - p["low"]) - (p["high"] - p["close"])) / rng
    vol = p["volume"].astype(float)
    return float((mfm.fillna(0) * vol).sum() / vol.sum()) if vol.sum() else 0.0


def sig_smart_money(inst_net: pd.Series | None, px: pd.DataFrame, cfg,
                    branch_ratio: float | None = None, branch_detail: str = "") -> dict:
    """主力買超的可行替代指標：
    1) 三大法人近 N 日淨買超佔成交量比、買超天數比
    2) Chaikin 資金流（CMF）：股價不動但收盤常落在日內高檔 → 有人在承接
    3) （選用）分點：前幾大買超分點的集中度
    """
    parts, details = [], []
    n = cfg["days"]
    if inst_net is not None and len(inst_net) >= n // 2:
        net = inst_net.iloc[-n:]
        vol = px["volume"].iloc[-len(net):].sum()
        ratio = net.sum() / vol if vol else 0
        pos_days = float((net > 0).mean())
        p = 0.6 * lin(ratio, cfg["target_net_ratio"], 0) + 0.4 * pos_days if ratio > 0 else 0.2 * pos_days
        parts.append((0.4, p))
        details.append(f"法人近{len(net)}日淨買 {net.sum() / 1000:+,.0f} 張（占量 {ratio:+.1%}，"
                       f"{pos_days:.0%} 天買超）")
    c = cmf(px, cfg["cmf_days"])
    parts.append((0.3, lin(c, cfg["target_cmf"], 0)))
    details.append(f"{cfg['cmf_days']}日資金流 CMF {c:+.2f}")
    if branch_ratio is not None:
        parts.append((0.3, lin(branch_ratio, cfg["target_branch_ratio"], 0)))
        details.append(branch_detail)
    w = sum(a for a, _ in parts)
    score = sum(a * b for a, b in parts) / w
    return res(score, "；".join(details), cmf=c, branch_ratio=branch_ratio)


def sig_insider(hist: pd.DataFrame | None, cfg) -> dict:
    """hist: columns ym, insider_shares（單一公司，依月份排序）。"""
    if hist is None or len(hist) < 2:
        return na("董監持股快照不足 2 個月（每月累積）")
    h = hist.sort_values("ym")
    cur, prev = h["insider_shares"].iloc[-1], h["insider_shares"].iloc[-2]
    first = h["insider_shares"].iloc[max(0, len(h) - 4)]
    if prev <= 0:
        return na("董監持股資料異常")
    chg = cur / prev - 1
    chg3 = cur / first - 1 if first > 0 else 0
    score = 0.6 * lin(chg, cfg["target_increase"], 0) + 0.4 * lin(chg3, cfg["target_increase"] * 2, 0)
    detail = f"董監持股月變動 {chg:+.2%}，近{min(len(h), 4) - 1}個月 {chg3:+.2%}"
    return res(score, detail, chg=chg, chg3=chg3)


# =====================================================================
# 三、基本面
# =====================================================================
def monthly_revenue(rev_df: pd.DataFrame) -> pd.Series:
    """FinMind TaiwanStockMonthRevenue → 以月份 Period 為索引的營收序列。"""
    if rev_df is None or rev_df.empty:
        return pd.Series(dtype=float)
    idx = pd.PeriodIndex([pd.Period(year=int(y), month=int(m), freq="M")
                          for y, m in zip(rev_df["revenue_year"], rev_df["revenue_month"])])
    s = pd.Series(rev_df["revenue"].astype(float).values, index=idx)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    full = pd.period_range(s.index.min(), s.index.max(), freq="M")
    return s.reindex(full)


def sig_turnaround(rev: pd.Series, cfg, per_latest: float | None = None) -> dict:
    """過去一兩年業績差：前 24~3 個月 YoY 多為負，或一年前的近12月營收比兩年前衰退。"""
    if len(rev.dropna()) < cfg["min_months"]:
        return na("月營收資料不足")
    yoy = rev / rev.shift(12) - 1
    past = yoy.iloc[-24:-2].dropna()
    neg_ratio = float((past < 0).mean()) if len(past) else 0
    ttm = rev.rolling(12).sum()
    ttm_drop = (ttm.iloc[-13] / ttm.iloc[-25] - 1) if len(ttm) >= 25 and ttm.iloc[-25] > 0 else np.nan
    p1 = lin(neg_ratio, 0.6, 0.2)
    p2 = lin(-ttm_drop, 0.20, 0) if not np.isnan(ttm_drop) else 0
    loss = per_latest is not None and per_latest <= 0
    score = max(p1, p2) * 0.85 + 0.15 * loss
    detail = f"過去兩年有 {neg_ratio:.0%} 月份營收年減"
    if not np.isnan(ttm_drop):
        detail += f"；一年前的近12月營收較前一年 {ttm_drop:+.0%}"
    if loss:
        detail += "；目前本益比為負/不適用（虧損中）"
    return res(score, detail, neg_ratio=neg_ratio, ttm_drop=ttm_drop, loss=loss)


def sig_yoy_turn(rev: pd.Series, cfg) -> dict:
    """營收 YoY 由負轉正。"""
    if len(rev.dropna()) < 15:
        return na("月營收資料不足")
    yoy = (rev / rev.shift(12) - 1).dropna()
    if len(yoy) < 4:
        return na("月營收資料不足")
    last, prev = yoy.iloc[-1], yoy.iloc[-6:-1]
    mom = rev.iloc[-1] / rev.iloc[-2] - 1 if rev.iloc[-2] else np.nan
    turned = last > 0 and prev.mean() < 0
    second = len(yoy) >= 2 and yoy.iloc[-2] > 0 and yoy.iloc[-7:-2].mean() < 0
    if turned:
        score = 1.0
    elif second and last > 0:
        score = 0.85  # 連兩月轉正
    elif last > 0:
        score = 0.4   # 本來就正成長，不算「由負轉正」
    else:
        score = 0.25 * lin(last - prev.mean(), 0.15, 0)  # 仍負但跌幅收斂
    score = min(1.0, score + (0.1 if mom > 0 else 0))
    ym = yoy.index[-1]
    detail = (f"{ym.year}/{ym.month:02d} 營收 YoY {last:+.1%}（前5月平均 {prev.mean():+.1%}），"
              f"MoM {mom:+.1%}")
    return res(score, detail, yoy=last, prev_avg=prev.mean(), mom=mom, turned=bool(turned or second))


def sig_theme(code: str, industry: str, themes: dict) -> dict:
    hits_code = [t for t, v in themes.items() if code in [str(x) for x in (v.get("codes") or [])]]
    if hits_code:
        return res(1.0, "題材清單：" + "、".join(hits_code), themes=hits_code)
    hits_ind = [t for t, v in themes.items()
                if any(i and i in (industry or "") for i in (v.get("industries") or []))]
    if hits_ind:
        return res(0.4, "產業別落在題材範圍：" + "、".join(hits_ind) + "（未經人工確認）",
                   themes=hits_ind)
    return res(0, "未列入題材清單")


# =====================================================================
# 四、市場心理與估值
# =====================================================================
def sig_valuation(per_df: pd.DataFrame | None, px: pd.DataFrame, years: int) -> dict:
    parts, details = [], []
    pos = price_position(px, years)
    parts.append((0.4, lin(pos, 0.1, 0.6)))
    details.append(f"股價位於 {years} 年區間 {pos:.0%}")
    pbr_pct = per = pbr = None
    if per_df is not None and not per_df.empty:
        p = per_df.sort_values("date")
        pb = p["PBR"].astype(float)
        pb = pb[pb > 0]
        if len(pb) > 60:
            pbr = float(pb.iloc[-1])
            pbr_pct = float((pb <= pbr).mean())
            parts.append((0.4, 0.7 * lin(pbr_pct, 0.1, 0.6) + 0.3 * lin(pbr, 0.8, 2.0)))
            details.append(f"PBR {pbr:.2f}（{len(pb) // 250 or 1} 年百分位 {pbr_pct:.0%}）")
        per = float(p["PER"].astype(float).iloc[-1])
        if per > 0:
            parts.append((0.2, lin(per, 10, 30)))
            details.append(f"PER {per:.1f}")
        else:
            details.append("PER 不適用（虧損）")
    w = sum(a for a, _ in parts)
    score = sum(a * b for a, b in parts) / w
    return res(score, "；".join(details), price_pos=pos, pbr=pbr, pbr_pct=pbr_pct, per=per)


def sig_media(news_n: int | None, ptt_n: int | None, cfg) -> dict:
    parts, details = [], []
    if news_n is not None:
        parts.append(lin(news_n, cfg["news_quiet"], cfg["news_loud"]))
        details.append(f"近{cfg['days']}日新聞約 {news_n} 則")
    if ptt_n is not None:
        parts.append(lin(ptt_n, cfg["ptt_quiet"], cfg["ptt_loud"]))
        details.append(f"PTT 股板近{cfg['days']}日 {ptt_n} 篇")
    if not parts:
        return na("未查詢媒體聲量")
    return res(sum(parts) / len(parts), "；".join(details), news=news_n, ptt=ptt_n)
