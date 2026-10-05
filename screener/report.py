"""輸出：docs/results.json（給網頁）、docs/results.md 與 GitHub Actions 摘要。"""
import json
import os
from pathlib import Path

DIM = {
    "籌碼面": ["big_holder", "margin", "smart_money", "insider"],
    "技術面": ["base", "volume_dry", "ma_squeeze"],
    "基本面": ["turnaround", "yoy_turn", "theme"],
    "心理面": ["media", "valuation"],
}
LABEL = {
    "base": "長期打底", "volume_dry": "窒息量", "ma_squeeze": "均線糾結",
    "big_holder": "大戶增持", "margin": "融資退場", "smart_money": "主力承接",
    "insider": "內部人增持", "turnaround": "業績谷底", "yoy_turn": "YoY轉正",
    "theme": "題材", "media": "低聲量", "valuation": "低基期",
}


def write(items, meta, docs_dir: Path):
    docs_dir.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "dimensions": DIM, "labels": LABEL, "items": items}
    (docs_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    lines = [f"# 埋伏股掃描結果 {meta['generated_at']}", "",
             f"股票池 {meta['universe']} 檔 → 技術面初選 {meta['stage1']} 檔 → "
             f"深入分析 {meta['stage2']} 檔 → 上榜 {len(items)} 檔", "",
             "| # | 代號 | 名稱 | 總分 | 資料覆蓋 | 點火 | 重點 |",
             "|---|---|---|---|---|---|---|"]
    for i, it in enumerate(items, 1):
        good = [LABEL[k] for k, v in it["signals"].items()
                if v.get("score") is not None and v["score"] >= 0.7]
        lines.append(f"| {i} | {it['code']} | {it['name']} | {it['score']:.1f} | "
                     f"{it['coverage']:.0%} | {'🔥' if it['trigger']['fired'] else ''} | "
                     f"{'、'.join(good)} |")
    if meta.get("warnings"):
        lines += ["", "## 執行警告", *[f"- {w}" for w in meta["warnings"]]]
    md = "\n".join(lines)
    (docs_dir / "results.md").write_text(md, encoding="utf-8")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
