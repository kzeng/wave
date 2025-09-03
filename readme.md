# 开盘挂单策略生成器

简要说明与用法。

## 安装依赖

需要 Python 3.9+ 与以下包：

- yfinance
- pandas
- numpy

## 运行

必需参数：

- --cash 可用资金（元）

可选参数：

- --limit 涨跌停幅度（默认 0.10）
- --offline 离线模式，仅使用本地缓存（data_cache/*.csv）

示例：

```cmd
python s1.py --cash 43380 --limit 0.1
```

若遇到数据源限流，可多试几次，或先运行一次联网成功后，后续用离线模式：

```cmd
python s1.py --cash 43380 --offline
```

脚本会在当前目录生成当日挂单 CSV：orders_YYYYMMDD.csv。

