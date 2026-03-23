# AI-Pedia Evaluation Rules & Rewrite Blueprint

## 0. 目的与范围
本文件定义 AI-Pedia 新版 evaluation 的统一规则、评分口径、输出字段与代码改造蓝图。

适用范围：
- 仅针对 evaluation 工作流（不依赖上传文件能力）
- 评估对象为系统生成阶段：Slides / Code / Quiz / Video 四产物
- 结果用于论文 Section 4.1 与 Section 5.x 的可复现数据支撑

不纳入本次范围：
- single-agent 外部产品对比（后续手动完成）
- token 预算限制（仅记录，不设 fail 阈值）

---

## 1. 已冻结的核心规则（必须遵守）

### 1.1 产物完整性硬门槛
每个 run 必须同时生成并可评估以下四个产物：
- Slides
- Code
- Quiz
- Video

若缺任一产物：
- `success_flag=False`
- `overall_score=0`

### 1.2 延迟硬门槛
- 只统计**系统生成阶段**（start -> workflow_complete）
- `latency_budget = 2400s`

若超过 2400s：
- `success_flag=False`
- `overall_score=0`

### 1.3 Silent Failure 判定
若满足以下条件则判定 silent failure：
- 超过 15 分钟无新事件
- 且未收到 `workflow_complete`

### 1.4 Token 规则
- 不设 token budget 上限
- 仅记录 `tokens_total / tokens_by_agent`
- token 统计包含重试消耗
- agent 维度仅统计：`coder / presentation / quiz / video`

### 1.5 Slide 结构硬规则
严格要求：
- `title` 页恰好 1 个
- `content` 页至少 1 个
- `summary` 页至少 1 个

（页数上下界与 layout compliance 细项在论文可手写，不作为当前必须编码项）

### 1.6 Code 执行规则
- 运行环境：当前项目 venv
- 允许自动联网安装依赖（全局安装到当前 venv）
- 代码执行超时：30 分钟
- 自动 pip install 单次安装超时：20 分钟

### 1.7 Quiz 规则
固定：
- 10 题
- 每题 4 个选项
- 单选（唯一正确答案）

### 1.8 Coverage 与 Out-of-Scope
- Coverage 仅以 `slides + code` 作为教学材料集合 `KM`（不含 video）
- `OutOfScope = 1 - Coverage`

### 1.9 Cross-Artifact Consistency
按 `S/C/Q/V` 六个 pair 的 Jaccard 相似度等权平均。

### 1.10 Video-Slide Synchronization
- 使用 video agent 返回的 narration `scripts`（每页一段）作为 `V`
- 按页对齐
- 若页数不一致：按最短长度对齐

---

## 2. Rookie Student 主评测设计（必须跑）

## 2.1 模型策略
三模型各跑一次，最后对可用模型求均值。
当前先使用占位符（后续替换真实 ID/API）：
- `ROOKIE_MODEL_1 = "PHI3_SMALL_PLACEHOLDER"`
- `ROOKIE_MODEL_2 = "QWEN2_7B_PLACEHOLDER"`
- `ROOKIE_MODEL_3 = "GEMMA_7B_PLACEHOLDER"`

调用失败策略：
- 每模型失败后重试 2 次
- 仍失败则跳过该模型，并记录 `model_call_failed`
- 若 3 个模型全部失败：`rookie_eval_failed`，该 run 判 fail（`success_flag=False, overall_score=0`）

## 2.2 四组测试（每 run 动态执行）
每次 run 动态改写一份 equivalent quiz（平行卷），并做四组评测：
1. 原始 quiz 学习前：`Acc_ori_pre`
2. 原始 quiz 学习后：`Acc_ori_post`
3. 平行 quiz 学习前：`Acc_eq_pre`
4. 平行 quiz 学习后：`Acc_eq_post`

说明：
- 平行卷“同难度”仅靠 prompt 约束，不做额外自动难度校验
- Rookie 约束依赖 system prompt（要求以“新手”视角答题）

## 2.3 兼容论文字段映射（保留）
为兼容 paper 字段，额外生成：
- `Accpre = Acc_ori_pre`
- `Accpost = Acc_ori_post`
- `Acccontrol = Acc_eq_pre`（默认映射；论文写作可后调）
- `ΔAcc = Acc_ori_post - Acc_ori_pre`
- `ΔAccnet = (Acc_ori_post - Acc_ori_pre) - (Acc_eq_post - Acc_eq_pre)`

---

## 3. Sanity Check（系统 prompt + 阈值校验）

以下规则用于识别 rookie 评测异常；分为 warning 与 fail。

### 3.1 预学习分数泄漏检查
对每个成功模型：
- 若 `Acc_ori_pre > 0.45` 或 `Acc_eq_pre > 0.45` -> `knowledge_leak_warning`
- 若 `Acc_ori_pre > 0.65` 或 `Acc_eq_pre > 0.65` -> `knowledge_leak_fail`

### 3.2 学习后退化检查
- 若 `Acc_ori_post < Acc_ori_pre - 0.05` -> `post_drop_fail`
- 若 `Acc_eq_post < Acc_eq_pre - 0.05` -> `post_drop_fail`

### 3.3 异常跃迁检查
- 若 `Δori = Acc_ori_post - Acc_ori_pre > 0.60` -> `abnormal_gain_fail`
- 若 `Δeq = Acc_eq_post - Acc_eq_pre > 0.60` -> `abnormal_gain_fail`

### 3.4 多模型方差检查
对同一指标（四组分数）计算跨模型标准差：
- `std > 0.25` -> `high_variance_warning`

### 3.5 Sanity Fail 对总结果影响
出现任何 `*_fail`：
- `success_flag=False`
- `overall_score=0`

Warning 不直接 fail，但必须记录到输出字段。

---

## 4. 指标定义（计算口径）

## 4.1 概念抽取（TF-IDF）
统一用于 `KS/KC/KQ/KV` 与覆盖、一致性、同步指标。

配置：
- n-gram: `1-2`
- `top_k = 5`
- 小写化
- 去停用词
- 词形还原（lemmatization）
- IDF 语料：**单次 run 内四产物文本联合语料**

文本源：
- `KS`: slides 文本
- `KC`: code 文本
- `KQ`: quiz 文本（包含 question + options + explanation）
- `KV`: video narration scripts（按页）

## 4.2 关键公式
- `AlignSC = |KS ∩ KC| / |KS ∪ KC|`
- `Coverage = |KQ ∩ KM| / |KQ|`，其中 `KM = KS ∪ KC`
- `OutOfScope = 1 - Coverage`
- `Consistency = mean(Jaccard(KA, KB))`，`A,B ∈ {S,C,Q,V}` 的 6 对
- `Sync = mean_i Jaccard(KS_i, KV_i)`，按页最短长度对齐

---

## 5. 成功 run 的 overall_score

仅对通过硬门槛与 sanity fail 检查的 run 计算：

`overall_score = 0.30*Learn + 0.10*Slide + 0.20*Code + 0.15*Quiz + 0.15*Consistency + 0.10*Sync`

各分项归一化到 `[0,1]`：
- `Learn = clip((max(0,Δori) + max(0,Δeq)) / (0.30*2), 0, 1)`
- `Slide = slide_structural_compliance`
- `Code = 0.70*code_exec_pass + 0.30*AlignSC`
- `Quiz = 0.50*quiz_format_validity + 0.50*Coverage`
- `Consistency = Consistency`
- `Sync = Sync`

注：失败 run 固定 `overall_score=0`（不参与成功分项计算）。

---

## 6. Reliability / Robustness / Concurrency 实验口径

### 6.1 任务组织
按用户确认：
- 25 个单任务
- 25 个并发批次（每批 2 请求）

说明：
- 评测单元数：50（25 single + 25 batch）
- 实际请求总数：75

### 6.2 系统级指标
至少输出：
- `completion_rate`
- `partial_failure_rate`
- `silent_failure_rate`
- `latency_mean`
- `tokens_total`
- `tokens_by_agent`

定义：
- `partial_failure`: 四个 sub-agent 中部分失败或未完成
- `silent_failure`: 15 分钟无事件且无 workflow_complete

---

## 7. 输出文件与字段规范

保留原有：
- `run_level_results.csv`
- `topic_best_results.csv`
- `aggregate_summary.json`

新增 paper-friendly 字段（至少）：
- `Accpre`
- `Accpost`
- `Acccontrol`
- `ΔAcc`
- `ΔAccnet`
- `AlignSC`
- `Coverage`
- `OutOfScope`
- `Consistency`
- `Sync`
- `completion_rate`
- `partial_failure_rate`
- `silent_failure_rate`
- `latency_mean`
- `tokens_total`
- `tokens_by_agent`

建议新增原子字段（便于后续灵活改论文口径）：
- `Acc_ori_pre`, `Acc_ori_post`, `Acc_eq_pre`, `Acc_eq_post`
- `Δori`, `Δeq`
- `rookie_models_success_n`, `rookie_failed_models`
- `sanity_warnings`, `sanity_fail_reasons`

---

## 8. 代码改造蓝图（按文件）

## 8.1 `evaluation/evaluation_runner.py`
目标：把当前 runner 升级为“硬门槛 + rookie 四组 + 系统级统计 + 新字段输出”。

主要改造点：
1. 运行阶段状态机
- 记录 `start_ts`
- 监听 SSE 事件并更新时间戳
- 加入 15 分钟 inactivity 检测（silent failure）
- 接收 `workflow_complete` 时结束

2. 硬门槛裁决
- 缺产物 -> fail
- latency > 2400 -> fail
- rookie_eval_failed -> fail
- sanity fail -> fail

3. rookie 流程接入
- 在产物评估后执行四组测试
- 三模型占位符调度
- 失败重试与跳过
- 均值聚合

4. 新评分流程
- 成功 run 才计算 `overall_score`
- 失败 run 强制 `overall_score=0`

5. 系统级实验批次控制
- 支持 25 single + 25 concurrent batch(2 each)
- 输出 batch/request 双维度统计

## 8.2 `evaluation/metrics.py`
目标：统一“确定性脚本 + TF-IDF 概念指标 + 同步指标 + 结构硬规则”。

主要改造点：
1. Slide 结构合规
- title=1 / content>=1 / summary>=1
- 输出 `slide_structural_compliance`

2. Code 执行
- timeout=30min
- 失败时支持依赖安装尝试（pip install timeout=20min）
- 记录安装日志摘要与是否恢复成功

3. Quiz 格式
- 固定 10 题 4 选 1
- 唯一答案合法性

4. TF-IDF 概念抽取模块
- 1-2gram, top_k=5, lowercase, stopword removal, lemmatization
- run 内四产物联合语料计算 IDF

5. 指标函数
- `compute_align_sc`
- `compute_coverage_out_of_scope`
- `compute_consistency_6pair`
- `compute_sync_pagewise`

## 8.3 `evaluation/token_logging.py`
目标：增强兼容性与 agent 粒度统计稳定性。

主要改造点：
1. 同时支持：`summary.json`、`trace_log.jsonl`、`trace.jsonl`
2. 同时兼容：
- 顶层 `total_tokens/tokens_by_agent`
- 嵌套 `tokens.total/tokens.by_agent`
- trace 中 `usage.total_tokens`
3. 聚合到统一格式：
- `tokens_total`
- `tokens_by_agent`（只保留 coder/presentation/quiz/video）
- `llm_call_count`

## 8.4 建议新增模块（可选但推荐）
- `evaluation/rookie_eval.py`：四组测试与模型聚合
- `evaluation/sanity.py`：sanity check 规则
- `evaluation/scoring.py`：overall_score 计算

（即使不拆文件，也要在逻辑层保持这三块职责分离）

---

## 9. 执行顺序（实施建议）
1. 先改 `metrics.py`（指标底座）
2. 再改 `token_logging.py`（观测兼容）
3. 最后改 `evaluation_runner.py`（流程编排与输出）
4. 跑小样本 dry-run（2 topics x 1 run）
5. 再跑完整实验批次（50 单元）

---

## 10. 占位配置（后续替换）
为保证现在即可开发，先保留配置占位：
- rookie model ids
- rookie provider / api endpoint
- 可选 stopword/lemmatization backend

替换时只改配置，不改评测主逻辑。

---

## 11. 最终约束（不可破坏）
- 缺产物 fail
- 超时 fail（2400s）
- silent failure fail（15min inactivity）
- rookie 全失败 fail
- sanity fail 触发 run fail
- 失败 run `overall_score=0`

