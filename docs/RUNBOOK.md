# 运维手册（RUNBOOK）

## 1. 启动与停止

### 本地（离线，零密钥）

```bash
pip install -r requirements.txt && pip install -e .
python -m uvicorn crossmind.api.app:app --host 127.0.0.1 --port 8000
# 控制台 http://127.0.0.1:8000/console/   文档 /docs   健康 /readyz
```

### 容器

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f crossmind
docker compose down          # 保留 var 卷
docker compose down -v       # 连数据一起删（不可恢复）
```

### 在线模型

```bash
export XM_LLM_PROVIDER=openai
export XM_LLM_BASE_URL=https://api.deepseek.com/v1
export XM_LLM_API_KEY=sk-...
export XM_LLM_MODEL=deepseek-chat
```

改完重启生效（配置在启动时校验，不合法会直接启动失败，不会带病运行）。

## 2. 健康检查

| 端点 | 含义 | 期望 |
|---|---|---|
| `/healthz` | 进程活着 | 200 `{"status":"ok"}` |
| `/readyz` | 索引已构建、可服务 | 200 且 `chunks > 0`；启动中返回 `starting` |
| `/metrics` | Prometheus 文本 | 可被抓取 |

容器 healthcheck 用的是 `/readyz`，因此"进程在但索引没建好"不会被判为健康。

## 3. 常用诊断

```bash
# 最近链路
curl -s localhost:8000/api/v1/ops/traces?limit=10 -H 'X-API-Key: demo-key-alpha'

# 单条 trace 的 span 树
curl -s localhost:8000/api/v1/ops/traces/<trace_id> -H 'X-API-Key: demo-key-alpha'

# 运行摘要（模式、语料、任务数、平均延迟）
curl -s localhost:8000/api/v1/ops/summary -H 'X-API-Key: demo-key-alpha'

# 只查召回（跳过生成，最快定位"是不是检索的锅"）
curl -s 'localhost:8000/api/v1/kb/search?q=保修期&top_k=5' -H 'X-API-Key: demo-key-alpha'
```

判读顺序：**先看检索分数 → 再看 trace 里 specialist 的耗时与 token → 最后看 critic 是否反复要求重写**。

## 4. 常见故障

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 401 `unauthorized` | 缺 `X-API-Key` 或 key 不在 `XM_API_KEYS` | 核对环境变量；key 以 `key:tenant` 形式配置 |
| 422 `guard_blocked` | 输入命中注入特征或超长 | 查看 `details.reason`；确认是误报则调 `InputGuard` 规则 |
| 429 `rate_limited` | 租户超过令牌桶 | 看 `details.retry_after_s`；临时调大 `XM_RATE_LIMIT_RPS` |
| 503 持续 | 在线模型不可用且熔断打开 | 看 `crossmind_llm_calls_total{status="degraded"}`；熔断 30s 后自动半开 |
| 拒答率突然升高 | 语料变更、阈值改动、embedding 服务异常 | 跑 `scripts/calibrate.py` 对比分布；检查 `retrieval` 的 `degraded` 标记 |
| 答案无引用 | 在线模型不遵守格式 | 系统会拦截为拒答并写 `warnings: ungrounded_answer_blocked`；检查提示词版本 |
| 任务卡在 `awaiting_approval` | 没人审批（设计如此） | 调用 `/tasks/{id}/approve`；超时策略见 Roadmap |
| 评测门禁失败 | hit@5 或拒答 < 0.9 | `cat reports/rag-eval.md` 看逐条失败，先定位是用例、检索还是生成 |

## 5. 指标含义速查

| 指标 | 异常含义 |
|---|---|
| `crossmind_http_request_duration_ms` p95 上升 | 先看是不是 SQLite 写竞争，再看 LLM 延迟 |
| `crossmind_llm_calls_total{status="degraded"}` > 0 | 主模型失败已回退，需查上游 |
| `crossmind_retrieval_total{outcome="no_answer"}` 占比升高 | 语料覆盖不足或阈值过高 |
| `crossmind_rate_limited_total` 升高 | 客户端突发或额度配置过低 |
| `crossmind_hitl_total{decision="pending"}` 增长 | 审批队列积压 |
| `crossmind_idempotent_replays_total` 升高 | 客户端重试频繁（可能是超时导致） |

## 6. 数据与备份

- 数据全部在 `var/`：`crossmind.db`（业务）、`traces/traces.jsonl`（链路）。
- 备份：停服后复制 `crossmind.db`（WAL 模式需同时复制 `-wal`/`-shm`，或先 `PRAGMA wal_checkpoint(TRUNCATE)`）。
- 语料与评测集在 `data/`，属于代码资产，随仓库版本管理。
- 报告在 `reports/`，可随时由脚本重建。

## 7. 密钥轮换

1. 在 `XM_API_KEYS` 中追加新 key（旧 key 保留），重启；
2. 通知调用方切换；
3. 观察旧 key 的请求量降为 0 后删除旧 key，再重启。

平台不存储用户密钥明文以外的任何凭证；日志只记录 key 的 8 位指纹。

## 8. 发版检查单

- [ ] `pytest tests -q` 全绿（145 条，覆盖率 ≥ 90%）
- [ ] `python scripts/run_eval.py` 退出码 0，指标不低于上一版
- [ ] `python scripts/load_test.py` 无错误，p95 不超过基线 20%
- [ ] `/readyz` 返回 `ready`
- [ ] 变更记录写入 `CHANGELOG.md`
- [ ] 若改动了提示词或阈值，在 PR 描述里贴出评测对比


