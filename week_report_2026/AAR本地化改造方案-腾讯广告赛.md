# AAR 本地化改造方案：把 automated-w2s-research 用于腾讯广告算法大赛

> 目标：把 Anthropic 的 `safety-research/automated-w2s-research`（下称 AAR）从"云端（RunPod + S3 + Anthropic API）跑对齐研究"改造成"在自有 5×8 H20 + Ceph 集群上，让 agent 自动迭代打腾讯广告算法大赛"。
> 资源：5 台服务器 × 8 卡 H20 = **40 张 H20**，多机共享 **Ceph** 存储。

---

## 〇、先认清两个事实（这决定了改造的重心）

### 事实 1：AAR 的 AWS 耦合其实很轻
读完 repo，它**并不重度依赖 AWS**：
- 算力走的是 **RunPod**（不是 EC2/Batch），封装在 `infrastructure/runpod.py`；
- 存储只用 **S3 兼容对象存储**（`infrastructure/s3_utils.py`），且已支持 `S3_ENDPOINT_URL` 自定义端点——意味着换成 MinIO/Ceph RGW 几乎零改动；
- 没用 Lambda / SQS / Bedrock；LLM 走 **Anthropic API**；
- 已内置三种执行模式：**Mode A 本地子进程 / Mode B 本地 Docker / Mode C RunPod 云**。

> **关键结论**：你不需要"去 AWS 化"，你真正要做的是两件事——① 把 Mode C 的 RunPod 编排换成"本地 40 卡调度"；② **把整个评估任务从 W2S/PGR 换成腾讯广告赛的赛题**。后者才是真正的工作量和亮点。

### 事实 2：大赛的评估范式和 AAR 天生同构
2025/2026 腾讯广告赛（全模态生成式推荐 / 统一序列建模）的规则：
- **提交训练+推理代码**，在官方评测环境里跑（不是提交预测文件）；
- **黑盒私有测试集**，ground truth 不下发；
- 排名指标：初赛 **HitRate@10 + NDCG@10**，2026 赛道用 **AUC**；复赛转化行为加权（2.5×）；
- 有**推理延迟预算**，超时的提交无效。

这跟 AAR 的"server 持有 ground truth、agent 只能提交东西换回一个分数"**完全是同一种闭环**。所以套用 AAR 是合理的——你只是把"PGR 打分器"换成"大赛式打分器"。

---

## 一、改造清单（按模块 + 工作量分级）

### ① 存储：S3 → Ceph（工作量：低）
两种做法，推荐前者：

**方案 A（最简）：Ceph 直接挂载为共享文件系统（CephFS）**
- 把 AAR 里所有"上传 S3 / 下载 S3"的产物路径，改成指向 CephFS 挂载点（如 `/mnt/ceph/w2s/`）。
- 因为 40 卡分布在 5 台机器，共享 FS 天然解决"worker 产物互相可见 + findings 同步"的问题（AAR 原本靠 S3 做这件事）。
- 改动点：`infrastructure/s3_utils.py` 抽象成一个 `Storage` 接口，实现一个 `LocalCephStorage`（就是普通文件读写 + 文件锁）。

**方案 B（改动更小但要多部署一个服务）：Ceph RGW / MinIO 提供 S3 兼容端点**
- 直接设 `S3_ENDPOINT_URL=http://<minio>:9000`，AAR 的 boto3 代码几乎不用改。
- 适合你想"尽量少动代码"时。

> 面试talking point：讲清"为什么选 CephFS 而不是保留 S3 语义"——因为本地多机共享 FS 让轨迹/findings 的跨 worker 同步从'显式上传下载'退化成'直接读目录'，少了一层一致性风险。

### ② 算力调度：RunPod → 本地 40 卡分配器（工作量：中）
这是替换 `infrastructure/runpod.py` 的核心。RunPod 做的事是"申请 pod → 塞镜像 → 跑 worker → 回收"，本地要自己实现"GPU 资源池 + 任务分发"。

**推荐：Ray（最贴合 AAR 的多 worker 异步模型）**
- 在 5 台机器上起一个 Ray 集群（1 head + 4 worker node），把 40 张 H20 注册为资源。
- 把 AAR 的"一个 experiment worker"包成 Ray task/actor，用 `@ray.remote(num_gpus=N)` 声明每个实验要几张卡。
- Ray 自动做调度、排队、失败重试——正好替代 RunPod 的 `MAX_CONCURRENT_PODS` 和"容量错误自动重试"。

**备选：Slurm**（如果你们机房已有 Slurm，直接 `sbatch` 提交，最省事，但异步编排要自己写轮询）。

**GPU 预算决策（重要，也是面试亮点）**：
- 40 卡怎么切？取决于单个实验要几张卡。若一个生成式推荐训练用 **4 卡**，则可并发 **10 个 agent 实验**；用 **8 卡（整机）** 则并发 5 个。
- 建议留 **1~2 张卡**给 LLM 网关/推理和评估服务，实际训练池按 ~36 卡规划。
- **本地算力是硬约束**（不像 RunPod 可弹性扩容）→ 策略要从"并行铺量"转向"更早剪枝无效方向"（这正是 AgentX 论文的洞察，可引用）。

### ③ 隔离 worker：Docker（工作量：低）
- AAR 的 Mode B（本地 Docker + GPU passthrough）**基本可直接用**，只需宿主机装 NVIDIA Container Toolkit。
- 保留它的"只挂载无标签 data/ + 只读 cache/"的防作弊设计——但在大赛场景里，"作弊"变成"agent 偷看测试集/ground truth"，同样要隔离。
- 镜像里的 CUDA 版本要匹配 H20（Hopper 架构，需 CUDA 12.x + 对应 PyTorch/vLLM 版本），这是唯一要注意的兼容点。

### ④ LLM 后端：Anthropic API → LLM 网关（工作量：中）
agent 大脑（提方案/写代码的 Claude）需要 LLM。两条路：

- **务实路线**：保留调用外部大模型 API，但前面加一个**本地 LLM 网关**（如 LiteLLM），做多 key 轮换、限流、失败重试、成本/调用量统计。AAR 原本假设 API 无限可用，本地要控成本和并发。
- **完全内网路线**：若比赛/合规要求数据不出内网，用你的 H20 部署一个开源大模型（如 Qwen/DeepSeek 系列）做 agent 大脑，vLLM 起推理服务，网关指过去。代价是 agent 的代码/推理能力可能下降，需要更强的提示工程和验证兜底。

> 决策点：这会占用你宝贵的 H20。建议初期用外部 API + 网关（省卡给训练），跑通后再评估是否内网化。

### ⑤ 评估层：W2S/PGR → 大赛评估适配层（工作量：高，核心）
这是把 AAR "变成打比赛的" 最关键一步。原来 `research_loop/tools/evaluate` + `core/eval.py` 算的是 PGR；你要替换成大赛口径：

1. **任务定义**：把 `ideas/` 下的"研究想法"重新定义为"推荐模型方案"——每个 idea 的 `run.py` 接收 config、产出一个可提交的训练+推理产物。
2. **本地评估器**：用大赛公开的 **baseline + 公开数据集（TencentGR-1M/10M，HuggingFace 有）** 搭一个**本地评测服务**，实现：
   - 从 agent 产物跑推理 → 在你**自己切出的验证集**上算 **HitRate@10 / NDCG@10 / AUC**（复赛加转化 2.5× 权重）；
   - **强制延迟预算检查**：模拟官方"超时即无效"，把推理耗时也作为一个硬门（这是很多队伍忽略的点，agent 要学会在精度和延迟间权衡）。
3. **ground truth 隔离**：验证集答案放评估服务端，agent 只能提交预测/代码换回分数——完整复刻 AAR 的防作弊闭环。
4. **和线上榜的一致性校准**：定期用一次真实提交校准"本地验证分 vs 线上榜分"的偏差，防止 agent 在本地过拟合（对应 AgentX 的"离线≠线上"问题）。

### ⑥ agent 任务模板/提示词：对齐研究 → 推荐调参（工作量：高）
- 改写 `research_loop/prompt.jinja2`：把领域知识从"weak-to-strong 对齐"换成"生成式推荐 / 序列建模 / 特征交互 / 语义 ID"，喂给 agent 大赛的赛题说明、baseline 结构、特征 schema。
- 扩充 `ideas/TEMPLATE`：给 agent 一批可迭代的"改进杠杆"——模型结构、embedding 用法、负采样、序列截断长度、多目标加权、蒸馏等。
- 加**领域护栏/验证**（借鉴 AgentX 的 Developing Agent）：agent 生成的训练代码要先过 DryRun（小样本能跑通、无 shape 错误、产物格式合法），再进真正训练，避免浪费 H20 跑注定崩的实验。

---

## 二、建议的落地顺序（先跑通，再优化）

| 阶段 | 目标 | 关键动作 |
|---|---|---|
| **P0 单机跑通** | 1 台机 8 卡，端到端闭环能转 | Ceph 挂载 + 改 evaluate 为大赛口径 + 单 agent 串行迭代 |
| **P1 多机并发** | Ray 集群拉起 40 卡，多 agent 并行 | 替换 runpod.py 为 Ray 调度 + GPU 预算切分 |
| **P2 可靠性** | 长跑不崩、失败自愈 | 故障分类（OOM/超时/数据错）+ 看门狗 + DryRun 前置门 |
| **P3 智能化** | agent 越迭代越准 | 轨迹/findings 复用、无效方向剪枝、本地-线上分校准 |

---

## 三、面试视角：这个项目怎么讲最值钱

即使是"沿用 AAR + 本地化"，你的**个人技术判断**全在改造决策里，重点讲这几条：

1. **"我发现 AAR 的 AWS 耦合其实很轻，真正的工作量是评估层的领域迁移"**——证明你读懂了系统本质，不是被表面框架吓住。
2. **"本地算力是硬约束，所以我把策略从铺量改成早剪枝"**——展示资源意识和权衡能力（可对标 AgentX 的洞察）。
3. **"我复刻了 ground truth 服务端隔离，并加了本地-线上分校准"**——展示你懂"防作弊 + 防过拟合"这类工业评估陷阱。
4. **"我给 agent 加了 DryRun 前置门，避免它把 H20 浪费在注定崩的实验上"**——展示可靠性工程思维。
5. **量化结果**：初赛/复赛稳定十几名 + 单 agent 并发数、一轮迭代耗时、迭代多少轮名次收敛、LLM 调用成本——**有数字才立得住**。

> 一句话故事线（面试开场可用）：
> "我把 Anthropic 的自动化研究 agent 框架本地化到我们的 40 卡 H20 集群上，替换掉它的云依赖，并把评估层从对齐研究改造成腾讯广告赛的生成式推荐赛题，让 agent 全自主地提方案、写代码、跑训练、按榜单指标迭代，最终在初赛和复赛都稳定拿到前 20。"

---

## 四、风险与注意点

- **H20 兼容性**：Hopper 架构，确认 Docker 镜像里 CUDA/PyTorch/vLLM/Unsloth 版本都支持，否则训练直接起不来。
- **Ceph 小文件性能**：agent 会产生大量小产物/日志，CephFS 对海量小文件可能有元数据压力，必要时聚合成 tar 或用对象存储模式。
- **LLM 成本失控**：多 agent 并行会疯狂调 LLM，网关的限流和预算上限必须先做，否则一夜烧光额度。
- **本地评估 ≠ 线上榜**：一定要做分校准，否则 agent 会在你的本地验证集上过拟合，线上翻车。
- **合规**：大赛数据脱敏但仍属腾讯业务数据，注意数据不出内网的要求，这会影响你 LLM 后端的选型。

---

*本方案基于 safety-research/automated-w2s-research README 与腾讯广告算法大赛 2025/2026 公开规则整理，供本地化改造与面试准备参考。具体字段以 repo 源码（core/config.py、core/eval.py、infrastructure/）与官方赛题为准。*
