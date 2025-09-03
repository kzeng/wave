#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开盘挂单策略生成器（A股）
------------------------------------------------
功能：
1) 自动从 Yahoo Finance 联网获取 1 年日线历史数据
2) 计算 6M/3M/1M/1W/1D 多周期动量 + ATR(14) 波动率
3) 基于“趋势 + 波动”生成当日开盘限价挂单（高抛低吸网格），给出清晰的价格与股数
4) 自动按 100 股一手与 10% 涨跌停价做边界与四舍五入

环境：
  pip install yfinance pandas numpy

用法示例：
  python open_order_strategy.py --cash 43380.31

备注：
- 证券代码需带交易所后缀：深市 .SZ，沪市 .SS
  例：000628.SZ（高新发展），600196.SS（复星医药）
- 若你更习惯写裸代码（如'000628'），可在 HOLDINGS 中用 'ticker_hint' 指定，
  程序会自动补全。
"""
import argparse
import dataclasses
import datetime as dt
import os
import math
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
import contextlib
import io
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from tqdm import tqdm


DEFAULT_RATE_WAIT_CEIL = 60  # seconds


def _looks_rate_limited(err: Exception) -> bool:
    msg = str(err) if err else ""
    msg_l = msg.lower()
    return (
        "rate limit" in msg_l or "too many requests" in msg_l or "429" in msg_l
    )



LOT_SIZE = 100        # A股一手 100 股
TICK_SIZE = 0.01      # 报价最小变动 0.01 元
LIMIT_PCT = 0.10      # 默认 10% 涨跌停幅度（ST 个股可改为 0.05）

# === 配置区（按需修改）========================================================
HOLDINGS = {
    # 你的现有持仓（来自 2025-09-03 截图）
    "000628.SZ": {
        "name": "高新发展",
        "shares": 500,
        "avg_cost": 96.367,
        "ticker_hint": "000628",     # 允许仅写数字代码，程序会补全 .SZ/.SS
        "buy_budget": 27000.0,       # 计划在本日用来“低吸”的资金上限（元）
    },
    "600196.SS": {
        "name": "复星医药",
        "shares": 200,
        "avg_cost": 26.991,
        "ticker_hint": "600196",
        "buy_budget": 16000.0,
    },
}
# ============================================================================


def infer_suffix(code: str) -> str:
    """如果传入的是裸代码，尽力推断交易所后缀。
    规则很简单：以 0/3 开头 -> .SZ；以 6 开头 -> .SS；否则原样返回。
    """
    if code.endswith(".SZ") or code.endswith(".SS"):
        return code
    if code[0] in ("0", "3"):
        return f"{code}.SZ"
    if code[0] == "6":
        return f"{code}.SS"
    return code


def _fetch_via_akshare(ticker: str, start_yyyymmdd: str) -> pd.DataFrame:
    """使用 AkShare 抓取日线数据，返回列：Open, High, Low, Close, Volume，索引为 Date。
    注意 AkShare 成交量单位为手，需乘以 100。
    """
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError("AkShare 未安装，请先 pip install akshare") from e

    code = ticker.split(".")[0]
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_yyyymmdd, end_date=None, adjust="")
    except Exception as e:
        raise RuntimeError(f"AkShare API 调用失败: {e}") from e
    
    if df is None:
        raise RuntimeError(f"AkShare 返回 None，可能股票代码 {code} 不存在或网络问题")
    
    if df.empty:
        raise RuntimeError(f"AkShare 返回空数据，股票代码: {code}，起始日期: {start_yyyymmdd}")

    # 调试信息：显示原始列名
    original_cols = list(df.columns)
    
    rename = {}
    cols = set(df.columns)
    if "日期" in cols: rename["日期"] = "Date"
    if "开盘" in cols: rename["开盘"] = "Open"
    if "最高" in cols: rename["最高"] = "High"
    if "最低" in cols: rename["最低"] = "Low"
    if "收盘" in cols: rename["收盘"] = "Close"
    if "成交量" in cols: rename["成交量"] = "Volume"
    
    if not rename:
        raise RuntimeError(f"AkShare 列名格式不识别，实际列名: {original_cols}")
    
    df = df.rename(columns=rename)
    need = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"AkShare 数据列不完整，缺少: {missing}，实际列: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])  # type: ignore
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # type: ignore
    # 成交量从手到股
    df["Volume"] = (df["Volume"].fillna(0) * 100).astype(float)

    df = df.sort_values("Date").dropna(subset=["Open", "High", "Low", "Close"])  # type: ignore
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].set_index("Date")  # type: ignore
    
    if df.empty:
        raise RuntimeError(f"AkShare 数据处理后为空，原始行数: {len(df) if df is not None else 0}")
    
    return df


def fetch_history(
    ticker: str,
    period_days: int = 180,  # 改为半年数据，更实用
    retries: int = 6,
    backoff: float = 2.0,
    offline: bool = False,
    rate_wait_ceil: int = DEFAULT_RATE_WAIT_CEIL,
    provider: str = "auto",  # auto|ak|yf
) -> pd.DataFrame:
    """获取历史日线，支持 AkShare/yfinance，带重试与本地缓存回退。

    缓存位置：脚本同目录下 data_cache/{ticker}.csv
    """
    cache_dir = Path(__file__).parent / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}.csv"

    # 离线模式：仅从缓存读取
    if offline:
        with tqdm(desc=f"读取缓存 {ticker}") as pbar:
            if cache_file.exists():
                df = pd.read_csv(cache_file, parse_dates=[0], index_col=0)
                df.index.name = "Date"
                pbar.set_description(f"✓ 缓存读取成功 {ticker}")
                return df
            pbar.set_description(f"✗ 缓存不存在 {ticker}")
        raise RuntimeError(f"离线模式且无缓存：{ticker}")

    start_date = dt.date.today() - dt.timedelta(days=int(period_days * 1.1))  # 稍微多取一点保证足够数据
    start_iso = start_date.isoformat()
    start_yyyymmdd = start_date.strftime("%Y%m%d")
    last_exc = None

    def save_cache(df: pd.DataFrame):
        try:
            df.to_csv(cache_file)
        except Exception:
            pass

    # 确定可用数据源；若未安装 akshare 则跳过 ak
    ak_available = True
    if provider in ("auto", "ak"):
        try:
            import akshare  # noqa: F401
        except Exception:
            ak_available = False

    try_order = []
    if provider == "auto":
        if ak_available:
            try_order.append("ak")
        try_order.append("yf")
    else:
        if provider == "ak" and not ak_available:
            # 直接跳过到 yfinance，相当于不可用
            try_order = ["yf"]
        else:
            try_order = [provider]
    
    # 对于 auto 模式，AkShare 失败 1 次后直接跳到 yfinance
    ak_failed = False
    for engine in try_order:
        # AkShare 失败后只尝试 1 次，然后立即切换
        engine_retries = 1 if (engine == "ak" and ak_failed) else (2 if engine == "ak" else retries)
        for attempt in range(1, engine_retries + 1):
            with tqdm(desc=f"获取数据 {ticker} ({'AkShare' if engine == 'ak' else 'yfinance'}) 尝试 {attempt}/{engine_retries}") as pbar:
                try:
                    if engine == "ak":
                        pbar.set_description(f"AkShare 获取 {ticker}...")
                        df = _fetch_via_akshare(ticker, start_yyyymmdd)
                    else:
                        pbar.set_description(f"yfinance 获取 {ticker}...")
                        # 尝试不同的 yfinance 参数组合
                        try:
                            # 方法1：标准方式
                            df = yf.download(
                                ticker,
                                start=start_iso,
                                interval="1d",
                                auto_adjust=False,
                                progress=False,
                                threads=False,
                            )
                            if df is not None and not df.empty:
                                pbar.set_description(f"yfinance 标准方式成功 {ticker}")
                            else:
                                # 方法2：尝试更短期间
                                pbar.set_description(f"尝试yfinance短期数据 {ticker}")
                                short_start = (dt.date.today() - dt.timedelta(days=90)).isoformat()
                                df = yf.download(
                                    ticker,
                                    start=short_start,
                                    interval="1d",
                                    auto_adjust=False,
                                    progress=False,
                                    threads=False,
                                )
                                if df is not None and not df.empty:
                                    pbar.set_description(f"yfinance 短期方式成功 {ticker}")
                                else:
                                    # 方法3：尝试 period 参数
                                    pbar.set_description(f"尝试yfinance period参数 {ticker}")
                                    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                        except Exception as yf_err:
                            raise RuntimeError(f"yfinance 所有方法都失败: {yf_err}")
                        
                        if df is None or df.empty:
                            # 输出调试信息
                            debug_info = f"ticker={ticker}, start={start_iso}, 结果=空"
                            raise RuntimeError(f"yfinance 返回空数据 ({debug_info})")
                        
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].dropna()
                    save_cache(df)
                    pbar.set_description(f"✓ {'AkShare' if engine == 'ak' else 'yfinance'} 获取成功 {ticker}")
                    return df
                except Exception as e:
                    last_exc = e
                    if engine == "ak":
                        ak_failed = True
                        # AkShare 失败后立即中断当前引擎的重试
                        pbar.set_description(f"✗ AkShare 失败，切换到 yfinance: {str(e)[:30]}...")
                        if provider == "auto":
                            print(f"AkShare 获取失败，立即切换到 yfinance: {ticker}")
                            break  # 跳出当前引擎的重试循环
                    
                    # 只在有剩余重试次数时等待
                    if attempt < engine_retries:
                        if _looks_rate_limited(e):
                            wait = min(15 * attempt, rate_wait_ceil)
                        else:
                            wait = (backoff ** attempt)
                        wait *= random.uniform(0.75, 1.5)
                        pbar.set_description(f"等待重试 ({wait:.1f}s) {str(e)[:50]}...")
                        time.sleep(wait)
                    else:
                        # 最后一次重试失败，记录错误但不等待
                        pbar.set_description(f"✗ {'AkShare' if engine == 'ak' else 'yfinance'} 失败: {str(e)[:30]}...")

    # 下载失败，尝试缓存回退
    with tqdm(desc=f"尝试缓存回退 {ticker}") as pbar:
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, parse_dates=[0], index_col=0)
                df.index.name = "Date"
                pbar.set_description(f"✓ 缓存回退成功 {ticker}")
                return df
            except Exception:
                pass
        pbar.set_description(f"✗ 缓存回退失败 {ticker}")
    raise RuntimeError(f"下载失败：{ticker}（{type(last_exc).__name__}: {last_exc}）")


def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """简易 ATR(14)"""
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = tr.rolling(n, min_periods=n).mean()
    return atr


@dataclasses.dataclass
class Metrics:
    close: float
    atr: float
    atr_pct: float
    sma20: float
    sma60: float
    ret_6m: float
    ret_3m: float
    ret_1m: float
    ret_1w: float
    ret_1d: float


def analyze(df: pd.DataFrame) -> Metrics:
    """计算一组用于决策的指标"""
    close = float(df["Close"].iloc[-1])
    atr = float(compute_atr(df).iloc[-1])
    sma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    sma60 = float(df["Close"].rolling(60).mean().iloc[-1])

    def pct_return(period: int) -> float:
        if len(df) <= period:
            return float("nan")
        return float(df["Close"].iloc[-1] / df["Close"].iloc[-period-1] - 1.0)

    ret_6m = pct_return(126)
    ret_3m = pct_return(63)
    ret_1m = pct_return(21)
    ret_1w = pct_return(5)
    ret_1d = pct_return(1)

    return Metrics(
        close=close,
        atr=atr,
        atr_pct=atr / close if close > 0 else float("nan"),
        sma20=sma20,
        sma60=sma60,
        ret_6m=ret_6m,
        ret_3m=ret_3m,
        ret_1m=ret_1m,
        ret_1w=ret_1w,
        ret_1d=ret_1d,
    )


def classify_bias(m: Metrics) -> str:
    """根据多周期动量 + 均线结构定义交易倾向"""
    votes = 0
    votes += 1 if m.ret_3m is not None and m.ret_3m > 0 else 0
    votes += 1 if m.ret_1m is not None and m.ret_1m > 0 else 0
    votes += 1 if m.close > m.sma20 else 0
    votes += 1 if m.sma20 > m.sma60 else 0

    if votes >= 3:
        return "bullish"   # 偏强：适度多卖、少买、网格步距略大
    if votes <= 1:
        return "bearish"   # 偏弱：多买、少卖、网格步距略小
    return "neutral"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def design_grid(m: Metrics, bias: str) -> Tuple[float, List[float], List[float]]:
    """根据波动率和倾向生成网格百分比（单步% 与 3 个层级）"""
    # 基础步距：与 ATR 占比相关，并限制在 [0.8%, 3%]
    base_pct = m.atr_pct
    if not (isinstance(base_pct, float) and math.isfinite(base_pct) and base_pct > 0):
        base_pct = 0.012
    base_step = clamp(base_pct * 0.9, 0.008, 0.03)

    if bias == "bullish":
        step = base_step * 1.15
    elif bias == "bearish":
        step = base_step * 0.9
    else:
        step = base_step

    # 三层网格（±1、±2、±3 个步距）
    buy_pcts = [-step * k for k in (1, 2, 3)]
    sell_pcts = [ step * k for k in (1, 2, 3)]
    return step, buy_pcts, sell_pcts


def round_price(p: float) -> float:
    return round(round(p / TICK_SIZE) * TICK_SIZE, 2)


def limit_bounds(last_close: float) -> Tuple[float, float]:
    up = round_price(last_close * (1 + LIMIT_PCT))
    dn = round_price(last_close * (1 - LIMIT_PCT))
    return dn, up


def tier_prices(last_close: float, pct_list: List[float]) -> List[float]:
    prices = [round_price(last_close * (1 + pct)) for pct in pct_list]
    dn, up = limit_bounds(last_close)
    return [clamp(p, dn, up) for p in prices]


def shares_per_tier(budget: float, prices: List[float]) -> List[int]:
    """按预算均分到每一层，按 100 股一手向下取整"""
    if budget <= 0:
        return [0 for _ in prices]
    per = budget / len(prices)
    lots = [int(per // (p * LOT_SIZE)) for p in prices]  # 以手为单位
    # 如果第一轮分配为 0，尝试把预算集中到更低价位
    if sum(lots) == 0:
        # 从最便宜的一层开始分配
        lots = [0 for _ in prices]
        for i in reversed(range(len(prices))):
            if budget >= prices[i] * LOT_SIZE:
                lots[i] = int(budget // (prices[i] * LOT_SIZE))
                budget -= lots[i] * prices[i] * LOT_SIZE
    return [x * LOT_SIZE for x in lots]


def sell_lots_available(held: int, tier_count: int = 3) -> List[int]:
    """把可卖持仓分成最多三档，按 100 股拆分"""
    max_sell_lots = held // LOT_SIZE
    plan = [0, 0, 0]
    for i in range(min(max_sell_lots, tier_count)):
        plan[i] = LOT_SIZE
    return plan


def build_orders_for_one(
    ticker: str,
    name: str,
    held_shares: int,
    buy_budget: float,
    offline: bool = False,
    retries: int = 6,
    rate_wait_ceil: int = DEFAULT_RATE_WAIT_CEIL,
    provider: str = "auto",
    period_days: int = 180,  # 新增参数
) -> pd.DataFrame:
    df = fetch_history(
        ticker,
        period_days=period_days,  # 传递参数
        offline=offline,
        retries=retries,
        rate_wait_ceil=rate_wait_ceil,
        provider=provider,
    )
    m = analyze(df)
    bias = classify_bias(m)
    step, buy_pcts, sell_pcts = design_grid(m, bias)

    last_close = m.close
    buy_prices = tier_prices(last_close, buy_pcts)
    sell_prices = tier_prices(last_close, sell_pcts)

    # 买入手数
    buy_shares = shares_per_tier(buy_budget, buy_prices)
    # 卖出手数（不超过当前可卖）
    sell_shares_raw = sell_lots_available(held_shares, 3)
    # 偏弱时减少卖出档位；偏强时保留 2~3 档
    if bias == "bearish":
        sell_shares_raw[0] = 0  # 偏弱少卖第一档，留待反弹更高再卖
    if held_shares < LOT_SIZE:
        sell_shares_raw = [0, 0, 0]

    comment = (
        f"bias={bias}, step≈{step*100:.2f}%, ATR%={m.atr_pct*100:.2f}%, "
        f"6M={m.ret_6m*100:.1f}%, 3M={m.ret_3m*100:.1f}%, 1M={m.ret_1m*100:.1f}%"
    )

    def rows(action: str, prices: List[float], shares_list: List[int]) -> List[Dict]:
        rows = []
        for i, (p, sh) in enumerate(zip(prices, shares_list), 1):
            if sh <= 0:
                continue
            rows.append({
                "date": dt.date.today().isoformat(),
                "ticker": ticker,
                "name": name,
                "action": action,
                "tier": i,
                "price": p,
                "shares": sh,
                "comment": comment,
                "cancel_by": "09:35",  # 集合竞价后 5 分钟不成交则撤
            })
        return rows

    sell_rows = rows("SELL", sell_prices, sell_shares_raw)
    buy_rows = rows("BUY", buy_prices, buy_shares)

    return pd.DataFrame(sell_rows + buy_rows)


def main():
    global LIMIT_PCT
    parser = argparse.ArgumentParser(description="根据历史数据生成 A股开盘挂单策略")
    parser.add_argument("--cash", type=float, required=True, help="可用资金（元）")
    # 注意：argparse 会把 % 视为占位符，需要写成 %% 才能在 --help 中显示百分号
    parser.add_argument("--limit", type=float, default=LIMIT_PCT, help="涨跌停幅度（默认10%%，ST可设为0.05）")
    parser.add_argument("--offline", action="store_true", help="离线模式：仅使用本地缓存，不联网")
    parser.add_argument("--retries", type=int, default=6, help="在线抓取失败的重试次数（默认6）")
    parser.add_argument("--rate-wait-ceil", type=int, default=DEFAULT_RATE_WAIT_CEIL, help="限流时单次最大等待秒数（默认60）")
    parser.add_argument("--provider", choices=["auto", "ak", "yf"], default="yf", help="历史数据来源（auto=优先AkShare，失败回退yfinance；yf=仅yfinance）")
    parser.add_argument("--period", type=int, default=180, help="历史数据期间（天数，默认180天即半年）")
    args = parser.parse_args()
    LIMIT_PCT = args.limit

    # 将裸代码补全后缀
    normalized = {}
    for k, v in HOLDINGS.items():
        t = infer_suffix(v.get("ticker_hint", k))
        normalized[t] = v
    # 预算总额不超过可用资金
    total_budget = sum(v["buy_budget"] for v in normalized.values())
    if total_budget > args.cash:
        scale = args.cash / total_budget
        for v in normalized.values():
            v["buy_budget"] *= scale

    all_orders = []
    failures = []
    for t, v in normalized.items():
        try:
            orders = build_orders_for_one(
                t,
                v["name"],
                v["shares"],
                v["buy_budget"],
                offline=args.offline,
                retries=args.retries,
                rate_wait_ceil=args.rate_wait_ceil,
                provider=args.provider,
                period_days=args.period,  # 传递期间参数
            )
            all_orders.append(orders)
        except Exception as e:
            failures.append((t, str(e)))
            print(f"跳过 {t}：{e}")

    if not all_orders:
        if failures:
            print("全部标的获取数据失败，请稍后重试或使用 --offline（需已有缓存）。")
        else:
            print("未生成任何挂单。")
        return

    result = pd.concat(all_orders, ignore_index=True)
    # 排序：先卖后买，按价格从近到远
    result["sort_key"] = result.apply(lambda r: (0 if r["action"]=="SELL" else 1, abs(r["price"]-result[result["ticker"]==r["ticker"]]["price"].median())), axis=1)
    result = result.sort_values(["action", "ticker", "tier"]).drop(columns=["sort_key"])

    # 输出
    today = dt.date.today().strftime("%Y%m%d")
    fname = f"orders_{today}.csv"
    cols = ["date","ticker","name","action","tier","price","shares","cancel_by","comment"]
    result[cols].to_csv(fname, index=False, encoding="utf-8-sig")

    print(f"\n=== 开盘挂单策略（{today}）===\n")
    print(result[cols].to_string(index=False))
    print(f"\n已导出: {fname}")
    print("\n执行建议：集合竞价 09:15-09:25 先挂全部限价单；09:35 未成交的按计划撤单或手动调整。\n")
    if failures:
        print("注意：以下标的因数据下载失败而跳过：")
        for t, msg in failures:
            print(f"- {t}: {msg}")


if __name__ == "__main__":
    main()
