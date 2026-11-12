# Changelog

遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本。

## [1.0.1] - 2026-11-12

对抗性探测（并发、超大输入、跨接口一致性）发现并修复的问题。

### Fixed

- **限流首个请求必 429**：`elapsed` 可能为负（`now` 采样早于桶的 `updated_at`），`burst == cost` 时首条请求被拒。
- **`/chat/stream` 拦截注入返回 500**：守卫在响应已开始后抛错，触发 `response already started`；改为流式响应前校验，现返回 422。
- **降级路径 TypeError**：`MockChatModel._render_generic` 参数不匹配，主模型故障回退时会二次崩溃。
- **`task.type` 恒为 `qa`**：路由出的意图只用于审计，未写回任务；现按意图持久化。
- **可答性门禁未在 `/chat` 生效**：图路径把检索结果直接交给生成，导致同一问题 `/chat` 作答、`/kb/answer` 拒答；门禁下沉到 `answer_with_context`。
- **并发重复 `Idempotency-Key` 建多个任务**：读后写存在竞态；改为原子预留（`INSERT OR IGNORE` + pending 标记），进行中返回 409。
- **超大请求体被接受**：新增 1 MB 上限与 413 错误信封。
- **审批后任务输出结构被替换**：只保留 `answer/state`，客户端字段断裂；改为在原输出上覆盖。
- **`XM_IDEMPOTENCY_TTL_S` 配置未生效**：现按 TTL 清理过期幂等记录。

### Changed

- `pyproject.toml` 的 `addopts` 去掉 `-q`，避免与命令行 `-q` 叠加成 `-qq` 导致汇总行消失。

## [1.0.0] - 2026-11-12

首个可交付版本：四模块业务闭环 + 生产级工程能力。

### Added

- **多智能体编排**：LangGraph `router → supervisor → research → specialist → critic → hitl_gate → finalize`，
  critic 带轮次上限的修订回路；Supervisor 输出计划与风险等级。
- **四类业务能力**：客服（RAG 问答）、文案（标题 + 五点）、翻译（术语表本地化）、选品（市场/竞品/建议三段报告）。
- **可溯源 RAG**：章节感知分块、BM25 + 特征哈希向量混合检索、饱和映射融合、确定性重排、
  `【n】` 引用系统回填、越界剔除、无引用即拒答。
- **可答性门禁**：jieba 分词 + IDF 加权覆盖，生成前拦截"实体缺失"型问题。
- **HITL**：对外发布任务暂停为 `awaiting_approval` 并持久化状态快照，审批后从快照恢复。
- **工程化**：API Key→租户、令牌桶限流、`Idempotency-Key` 幂等、统一错误码、输入防护（注入/PII）、
  模型重试/熔断/降级、JSON 日志、全链路 Trace、Prometheus 指标、审计表。
- **控制台**：零构建原生 JS，SSE 流式对话、步骤时间线、引用卡片、任务审批、检索调试、运维观测。
- **质量体系**：145 条测试（行覆盖率 92%）；26 条评测集与 6 项指标；标定脚本；压测脚本；CI 质量门禁。
- **交付物**：Dockerfile（非 root + healthcheck）、docker-compose、GitHub Actions、
  README / ARCHITECTURE / EVALUATION / RUNBOOK / ADR / RESUME。

### Known limitations

- 离线模型的翻译为术语替换夹具，不代表翻译质量。
- 可答性门禁为词面代理，"加拿大清关""加密货币付款"两条 near-miss 仍会误答（见 EVALUATION 第 5 节）。
- 单进程 SQLite 写是并发瓶颈（20 并发 p95 ≈ 200 ms）。
- 会话记忆仅最近 12 轮，无长期偏好记忆。

## [Unreleased]

- NLI / LLM judge 替代词面门禁。
- 写路径异步化与 Postgres 适配。
- 真实 reranker 接入与在线模型评测基线。

