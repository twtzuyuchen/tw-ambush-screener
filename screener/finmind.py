"""FinMind API 客戶端（逐檔查詢；內建節流，避免超過免費/會員額度）。"""
import time

import pandas as pd

from .net import SESSION, log

API = "https://api.finmindtrade.com/api/v4/data"


class QuotaExceeded(Exception):
    pass


class FinMind:
    def __init__(self, token, min_interval: float):
        self.token = token or None
        self.min_interval = min_interval
        self._last = 0.0
        self.calls = 0
        self.fail_count = {}
        self.disabled = set()

    def _throttle(self):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def get(self, dataset, data_id=None, start_date=None, end_date=None) -> pd.DataFrame:
        if dataset in self.disabled:
            return pd.DataFrame()
        params = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        self._throttle()
        self.calls += 1
        try:
            r = SESSION.get(API, params=params, headers=headers, timeout=60)
            j = r.json()
        except Exception as e:  # 網路錯誤：不中斷整體流程
            log.warning("FinMind %s %s 失敗：%s", dataset, data_id, e)
            return pd.DataFrame()
        status = j.get("status", r.status_code)
        if status == 402:
            raise QuotaExceeded(j.get("msg", "FinMind 額度用盡"))
        msg = str(j.get("msg", ""))
        if status != 200 and "level" in msg.lower():
            log.warning("FinMind %s 需要更高會員等級，本次執行停用（請在 config.yaml 關閉）", dataset)
            self.disabled.add(dataset)
            return pd.DataFrame()
        if status != 200:
            n = self.fail_count[dataset] = self.fail_count.get(dataset, 0) + 1
            log.warning("FinMind %s %s → %s %s", dataset, data_id, status, j.get("msg"))
            if n >= 3:  # 多半是權限不足（贊助限定資料集），停用以節省額度
                log.warning("FinMind %s 連續失敗，本次執行停用", dataset)
                self.disabled.add(dataset)
            return pd.DataFrame()
        self.fail_count[dataset] = 0
        return pd.DataFrame(j.get("data", []))
