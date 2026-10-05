"""離線測試：用合成資料驗證訊號與整條流程（不連網）。執行：python -m pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

from screener import main, signals as S

RNG = np.random.default_rng(7)
DAYS = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=750)


def make_px(kind):
    n = len(DAYS)
    if kind == "ambush":  # 先跌深，再一年窄幅箱型，量縮、均線收斂
        c = np.r_[np.linspace(80, 42, n - 260), 40 + 1.5 * np.sin(np.linspace(0, 12, 260))]
        c[-60:] = 41 + np.linspace(-0.3, 0.3, 60) + RNG.normal(0, 0.15, 60)
        v = np.r_[np.full(n - 260, 2.0e6), np.full(200, 6e5), np.full(60, 2.5e5)]
    elif kind == "trend":
        c = np.linspace(30, 120, n)
        v = np.full(n, 3e6)
    else:
        c = 50 * np.exp(np.cumsum(RNG.normal(0, 0.03, n)))
        v = np.full(n, 1e6)
    v = v * (1 + RNG.normal(0, 0.1, n)).clip(0.5)
    c = pd.Series(c, index=DAYS)
    df = pd.DataFrame({"open": c * 0.998, "high": c * 1.01, "low": c * 0.99, "close": c,
                       "volume": v}, index=DAYS)
    df.loc[df.index[-60:], "close"] = df["high"].iloc[-60:] * 0.999  # 收在高檔 → CMF 為正
    return df


PX = {"1111": make_px("ambush"), "2222": make_px("trend"), "3333": make_px("noise")}


def fake_finmind(self, dataset, data_id=None, start_date=None, end_date=None):
    self.calls += 1
    if dataset == "TaiwanStockMarginPurchaseShortSale":
        b = np.linspace(3000, 1500, 200) if data_id == "1111" else np.full(200, 2000.0)
        return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in DAYS[-200:]],
                             "MarginPurchaseTodayBalance": b})
    if dataset == "TaiwanStockInstitutionalInvestorsBuySell":
        rows = []
        for d in DAYS[-30:]:
            rows.append({"date": d.strftime("%Y-%m-%d"), "name": "Investment_Trust",
                         "buy": 30000 if data_id == "1111" else 0, "sell": 5000})
        return pd.DataFrame(rows)
    if dataset == "TaiwanStockMonthRevenue":
        per = pd.period_range(end=pd.Period.now("M") - 1, periods=40, freq="M")
        base = np.r_[np.linspace(100, 60, 28), np.linspace(60, 75, 12)]
        return pd.DataFrame({"revenue_year": per.year, "revenue_month": per.month,
                             "revenue": base * 1e6})
    if dataset == "TaiwanStockPER":
        return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in DAYS],
                             "PER": 0.0, "PBR": np.linspace(2.5, 0.9, len(DAYS))})
    if dataset == "TaiwanStockHoldingSharesPer":
        rows = []
        for k, d in enumerate(pd.date_range(end=DAYS[-1], periods=8, freq="W-FRI")):
            rows += [{"date": d.strftime("%Y-%m-%d"), "HoldingSharesLevel": "more than 1,000,001",
                      "people": 20, "percent": 40 + 0.3 * k},
                     {"date": d.strftime("%Y-%m-%d"), "HoldingSharesLevel": "total",
                      "people": 10000 - 150 * k, "percent": 100}]
        return pd.DataFrame(rows)
    return pd.DataFrame()


@pytest.fixture
def offline(monkeypatch, tmp_path):
    uni = pd.DataFrame({"code": list(PX), "name": ["埋伏", "飆股", "亂跳"],
                        "market": "twse", "industry": ["半導體業", "航運業", "食品工業"]})
    monkeypatch.setattr(main, "load_universe", lambda fm: uni)
    monkeypatch.setattr(main.prices, "download_prices", lambda u, p, c: PX)
    monkeypatch.setattr(main.official, "update_tdcc", lambda d: None)
    monkeypatch.setattr(main.official, "update_insider", lambda d: None)
    monkeypatch.setattr(main.FinMind, "get", fake_finmind)
    monkeypatch.setattr(main.sentiment, "lookup", lambda n, c, d: (1, 0))
    real = main.load_yaml

    def cfg_patch(p):
        c = real(p)
        if str(p).endswith("config.yaml"):
            c["paths"] = {"data": str(tmp_path / "data"), "docs": str(tmp_path / "docs")}
            c["universe"]["min_avg_lots"] = 0
            c["finmind"]["use_holding_shares"] = True
        return c
    monkeypatch.setattr(main, "load_yaml", cfg_patch)
    monkeypatch.setattr(main, "ROOT", main.ROOT)
    return tmp_path


def test_technical_signals():
    cfg = main.load_yaml(main.ROOT / "config.yaml")["technical"]
    a, t = PX["1111"], PX["2222"]
    assert S.sig_base(a, cfg["base"])["score"] > 0.6
    assert S.sig_base(t, cfg["base"])["score"] == 0
    assert S.sig_volume_dry(a, cfg["volume"])["score"] > 0.6
    assert S.sig_ma_squeeze(a, cfg["ma"])["metrics"]["spread"] < 0.04


def test_pipeline(offline):
    items, meta = main.run()
    assert items and items[0]["code"] == "1111", [i["code"] for i in items]
    top = items[0]
    for k in ("big_holder", "margin", "smart_money", "yoy_turn", "turnaround", "valuation"):
        assert top["signals"][k]["score"] is not None, k
    out = json.loads((offline / "docs" / "results.json").read_text(encoding="utf-8"))
    assert out["items"][0]["code"] == "1111" and out["items"][0]["group"] == "半導體業"
    print(json.dumps(top["signals"], ensure_ascii=False, indent=1)[:2500])


def test_yoy_turn():
    per = pd.period_range("2023-01", periods=30, freq="M")
    rev = pd.Series(np.r_[np.full(12, 100.0), np.full(12, 80.0), np.full(5, 78.0), [84.0]], index=per)
    r = S.sig_yoy_turn(rev, {})
    assert r["metrics"]["turned"] is True and r["score"] >= 1.0


def test_yfinance_split():
    from screener.prices import _split
    idx = pd.date_range("2025-01-01", periods=3)
    cols = pd.MultiIndex.from_product([["1111.TW", "2222.TWO"],
                                       ["Open", "High", "Low", "Close", "Volume"]])
    data = pd.DataFrame(np.ones((3, 10)), index=idx, columns=cols)
    data[("2222.TWO", "Close")] = np.nan
    out = _split(data, ["1111.TW", "2222.TWO"])
    assert list(out) == ["1111.TW"] and list(out["1111.TW"].columns)[3] == "close"


def test_industry_cap():
    assert main.primary_industry("電子工業、半導體業") == "半導體業"
    assert main.primary_industry("") == "其他"
    rows = [{"code": str(1000 + i), "group": "半導體業" if i < 15 else "航運業", "score": 100 - i}
            for i in range(20)]
    out = main.cap_by_industry(rows, 10, key=lambda r: r["score"])
    assert sum(r["group"] == "半導體業" for r in out) == 10
    assert sum(r["group"] == "航運業" for r in out) == 5
    assert [r["score"] for r in out] == sorted([r["score"] for r in out], reverse=True)
