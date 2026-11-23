# DiscoDemo 与 DemoGen：面向 Sim2Real 的轨迹多样性生成——方法、实验及可复现性深度调研

> **更新：2026-11-23** ｜**主题：轨迹多样性、模仿学习、数据合成、Sim2Real**｜**证据范围：原论文 + 官方项目 + GitHub 静态代码审计**  
> 本文**独立新增**，不替换仓库已有的完整综述与第四章。参考原有[轨迹多样性章节](chapters/04-diversity.md)阅读。
>
> **重要说明**：论文摘要部分为依据原文的**完整要点中文译述**，非逐词逐句对照译本；实验结果均为**作者报告值**，而非本研究运行所得。源码“存在”不等于环境已验证、“有真机实验”不等于完整真机复现已开放。文中标为**审计疑点**之处需要运行或作者澄清。

## 研究定位与先行结论

| 维度 | DiscoDemo | DemoGen |
|---|---|---|
| 论文 | *DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning* | *DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning* |
| 时间/状态 | 2026-10-23，arXiv v1；**未核实正式录用** | RSS 2025 正式发表 |
| 核心问题 | **同一任务发现多种成功解**，提高成功轨迹的覆盖度 | **一条示教迁移至大量新物体位姿**，提高空间泛化 |
| 方法类别 | 特权状态 RL teacher → 合成示教 → 视觉 VLA student | 对象中心运动/技能分段 + TAMP + 点云 3D 编辑 → DP3 |
| 数据产出 | 从仿真闭环执行中得到新轨迹、真实渲染观测和动作 | 对真实轨迹进行规则化空间变换，合成点云和动作 |
| 原始示教 | 每任务 20 条 | 原理以每任务 1 条为起点；真机实验为减轻感知噪声，**额外重放两次** |
| 真机证据 | FR3 四任务；π0.5 无额外真机微调 | Panda、灵巧手、双臂；8 项主任务及扰动/避障扩展 |
| 资源现状 | 主流程、种子数据、生成器/策略权重、评测与分析脚本公开 | 主生成算法、示例 Zarr、DP3/DP 训练及指定硬件部署代码公开 |
| 难点 | Isaac Sim 5.0/Isaac Lab 2.2 图形后端、真实场景标定、模型训练 | 点云对象分割、手工任务分段、较旧依赖、合成点云可见性误差 |
| 研究推荐 | **有 RTX 仿真节点时适合主线**；仅 H100 时先从已发布数据着手 | **低成本可落地基线**，尤其适合改进空间轨迹增强 |

两者并不直接解决同一个优化问题：DiscoDemo 侧重**行为模式（solution-mode）多样性**；DemoGen 侧重**初始配置（spatial-configuration）多样性**。从宽泛的“Sim2Real 数据生成”角度可以比较，从严格的“动力学模拟训练→真实部署”角度，DiscoDemo 更直接，DemoGen 更接近**real-demo-to-synthetic-data-to-real-policy**。

---

# 第一篇：DiscoDemo

**原题**：DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning  
**作者**：Minho Park、Kinam Kim、Donghu Kim 等；KAIST AI / Holiday Robotics。  
**发表记录**：2026-10-23 arXiv:2610.32093v1（以本次审计能核实的版本为准）。  
**原文**：[arXiv 摘要](https://arxiv.org/abs/2610.32093)｜[HTML 全文](https://arxiv.org/html/2610.32093v1)｜[作者项目](https://davian-robotics.github.io/DiscoDemo/)｜[GitHub](https://github.com/DAVIAN-Robotics/DiscoDemo)｜[HF 资源](https://huggingface.co/collections/DAVIAN-Robotics/discodemo-6ac696e70d028fddf0c3dfeb)

## 1. 摘要翻译（依据原文 Abstract 的中文译述）

机器人操作中的行为克隆高度依赖训练数据的覆盖范围，而收集各种**不同且成功**的示教仍需付出高昂成本。本文不将强化学习作为最终要部署的控制器，而是用它为下游视觉—运动模仿学习**生产训练数据**。基本判断是：相较反复收集同一种有限的成功策略，收集多种不同的成功解更有机会覆盖广泛的状态与动作轨迹。

为此，作者提出 DiscoDemo：使用具备仿真特权状态访问能力、由潜变量控制的强化学习生成器，并结合并行逆向课程学习及**仅对成功轨迹生效的多样性目标**，学习多种完成任务的行为。训练完成后，让生成器产生多条成功闭环轨迹，对不理想的物理交互进行安全过滤，再导出供视觉策略训练的静态数据集。

在四项操作任务上，方法生成的数据具有更广泛的成功状态覆盖，并改善了仿真和真实机器人上的视觉操作策略表现。作者报告：仿真平均成功率从最强对照的 **40% 提高到 62%**，真实机器人平均成功率从 **39% 提高到 78%**。论文据此指出，多种成功行为对预训练 flow-matching VLA 的模仿学习有益，且能够和在线纠错数据收集方法互补。[D1]

**一句话解读**：作者不是训练一个“可直接上真机的多技能 RL controller”，而是训练一个**在仿真里能发现多条不同成功解的数据生成器**，随后让视觉学生学习这些轨迹。

## 2. 架构图/方法图 + 核心技术与算法

### 2.1 作者图：真实工作台与方法总览

![DiscoDemo 作者仓库原图：方法示意与多样化轨迹](https://raw.githubusercontent.com/DAVIAN-Robotics/DiscoDemo/main/assets/teaser.png)

*作者仓库 teaser 原图；完整算法总览以原论文 [Figure 2](https://arxiv.org/html/2610.32093v1#S3) 为准。上图来自作者公开资产，非本文重绘。*

![DiscoDemo 作者图：真实示教与仿真工作台](https://raw.githubusercontent.com/DAVIAN-Robotics/DiscoDemo/main/assets/real2sim.jpg)

*作者仓库 real2sim 原图：实物与仿真场景、相机视角对齐示例。*

### 2.2 自绘中文算法架构

~~~mermaid
flowchart TD
  A["20条真实示教/任务"] --> B["Real-to-Sim：标定 FR3、相机、物体和延迟"]
  B --> C["Isaac Lab：可恢复示教中间状态"]
  C --> D["Parallel RFCL：从成功末端向起点逆向学习"]
  Z["每个Episode采样潜变量 z"] --> E["特权状态策略 pi_Gen(s,z)"]
  D --> E
  E --> F["FlashSAC 经验回放与更新"]
  F --> G["任务成功奖励 + 成功门控 METRA 多样性奖励 - 安全惩罚"]
  G --> E
  E --> H["固定生成器：原始初态分布并行 Rollout"]
  H --> I["成功判定 + 物理接触/夹具位移过滤"]
  I --> J["渲染相机 RGB + 保存动作 / LeRobot v3"]
  J --> K["冻结特权接口；π0.5 视觉/本体感知 SFT"]
  K --> L["仿真50次评测 + FR3真实机器人评测"]
~~~

### 2.3 Parallel Reverse Curriculum：稀疏奖励下的有效学习

- **为什么需要**：原始初态离目标很远，成功回报稀少，随机探索极难学习插孔或抓取。
- **怎么做**：每条示教有一条独立课程边界 c_i；环境从成功示教靠后状态初始化，随着该边界起始 episode 的成功率达标，把起点逐步前移。
- **关键配置**：每任务 20 条示教、2048 并行环境、逆向步长 16、成功阈值 0.90、最近 10 个边界 episode 统计；至少 50% 示教已从起点解决任务后切到自然初态分布。[D2]
- **工程落地**：**libs/FlashSAC/flash_rl/rfcl/reverse_curriculum_isaac.py** 和 **curriculum_core.py**；测试见 **tests/test_curriculum.py**。
- **理解边界**：示教不仅提供行为克隆样本，更提供能被仿真器直接 reset 的**状态序列**；没有可重置状态或精确仿真，该算法不可能原样移植到任意视频数据。

### 2.4 Success-Gated METRA：只在成功行为之间追求差异

生成器为每个 episode 采样 z ~ N(0,I)，论文实现的 latent dim 为 3，整个 episode 保持 z 不变。策略以 (s,z) 输出动作；同时训练一个表征网络 phi(s)，以相邻状态表征差分与潜变量的内积定义多样性回报：

$$
r_{\mathrm{div}}(s_t,s_{t+1},z)
=\left\langle\phi(s_{t+1})-\phi(s_t),z\right\rangle.
$$

结合二值轨迹成功标签 y(tau)，核心奖励是

$$
\tilde r_t=r_{\mathrm{task}}+\alpha\,y(\tau)\,r_{\mathrm{div}}-r_{\mathrm{safe}}.
$$

其中 alpha=0.3 是公开 DiscoDemo 配置；**P-RFCL** 对照配置设置 alpha=0、z_dim=0。METRA 使用表征学习与拉格朗日约束防止任意放大表示距离。独到之处是 **y(tau) 必须等整条轨迹结束后才可知**：代码在 replay transition 中加入 episode UID，使用 **SkillGateTable** 延迟登记成功，再在采样更新时补充 intrinsic reward。代码：[metra.py](https://github.com/DAVIAN-Robotics/DiscoDemo/blob/main/libs/FlashSAC/flash_rl/agents/utils/metra.py)、[skill_z.py](https://github.com/DAVIAN-Robotics/DiscoDemo/blob/main/libs/FlashSAC/flash_rl/envs/skill_z.py)。[D1][D2]

**与简单 trajectory perturbation 的差别**：z 并非直接加到末端轨迹上的噪声，而是由 RL 探索并学出不同成功解；多样性的实际收益仍需依靠学生模型评估，不能凭轨迹散点图判定。

### 2.5 双层安全设计：训练 shaping 与离线过滤

训练阶段通过非法接触力以及插接卡死惩罚，使生成器倾向于更安全的策略；生成阶段先筛掉失败轨迹，再依据仿真器接触/位移信号拒绝不合格的成功轨迹。

当前 **collision_metrics.py** 的导出阈值为：单步非法接触峰值超过 50 N；连续 3 个控制步的最小接触力超过 20 N；插孔任务卡死指标超过 15 N；容器平面最大位移达到 0.01 m。训练惩罚与导出硬过滤**并不等同**，触发物体、允许接触对由任务配置决定。[D3]

**审计疑点（需要补证）**：论文 Table 1 报告“过滤后”PnP-Banana 的平均容器位移约 1.04 cm、FMB-SqCircle 约 5.89 cm，但当前导出代码以单条轨迹最大平面位移 **>= 1 cm** 拒绝。若两者使用同一统计定义及过滤集，则均值不可能超过 1 cm；应检查论文实验版代码、是否采用不同参数或指标、真实导出结果，**不能直接宣称作者的安全统计已被源码证实**。[D4]

### 2.6 多样性指标与训练目标的区别

论文在分析阶段使用 Vendi Score，分别估计同一固定起始状态的 **solution diversity** 与跨初态分布的 **coverage**；由物体和末端位姿轨迹构造 embedding，经 DCT 时序压缩和 RBF kernel 获得相似度，再对归一化核矩阵谱做熵计算。Vendi 是**离线评估指标**，不应误认为 RL 直接以 Vendi 作为奖励优化。[D1]

## 3. 完整 Pipeline：从真实示教到学生策略

| 阶段 | 输入 | 实际处理 | 输出/代码入口 | 可单独验收的结果 |
|---|---|---|---|---|
| 0 Real-to-Sim | FR3、相机标定、对象、真实示教 | 任务场景与接触模型、示教状态重放 | **libs/real2sim/**、**data/demo_banks/** | 仿真初始状态、示教回放一致性 |
| 1 RL generator | 20 示教状态、可复位场景 | FlashSAC + RFCL + METRA + 训练安全惩罚 | **scripts/1_rl/train.sh** | 能输出 actor/normalizer、记录成功率与课程 |
| 2 Rollout | Stage 1 actor、真实初态分布 | 随机 z、多环境闭环采集 | **libs/RoboLab/scripts/collect_rl_states.py** | 成功轨迹 HDF5 与成功采集率 |
| 2 Filtering | 原始 HDF5、接触力及夹具状态 | 物理过滤 + 固定种子抽样 | **score_collision.py**、**select** 子命令 | keep.json、过滤率、安全指标 |
| 2 Rendering | 保留的仿真状态 | 多相机 RGB、动作标签、LeRobot v3 | **scripts/2_gendata/generate.sh** | 3,000 条/任务的静态策略数据 |
| 3 SFT | 公开 π0.5/DROID 基座与新数据 | BF16、动作增量标签、20k steps | **scripts/3_sft/train.sh**、**sft/action_delta.py** | 新 policy checkpoint、训练日志 |
| 4 仿真评测 | Stage 3 checkpoint | 每任务 50 episode、时间/成功率 | **python -m sft.sim_eval** | result.json、视频 |
| 5 真机 | 相机、FR3 控制系统 | 无额外真实微调直接部署 | 论文真机协议；专用端到端部署脚本未核实 | 真实任务成功率，不能只用仿真替代 |

官方四项任务：PnP-Banana、StackCube、FMB-Round、FMB-SqCircle；前二者时长 30 s，后二者 20 s。作者将 RL 生成阶段与下游 π0.5 学生训练分开，因而可以只重新采集生成数据，也可以固定 Stage 2 数据独立比较学生训练方法。[D1][D2]

## 4. 可复现性审查（GitHub 静态代码审计）

### 4.1 代码、数据与复现入口核实

- **主仓库**：<https://github.com/DAVIAN-Robotics/DiscoDemo>；本次检查的提交 **362b6f019c41e118d14fda0eeebde67bee2b916d**。
- **种子数据**：四个 **data/demo_banks/{task}/demos.h5** 文件在 Git tree 中存在，并非只有下载说明。
- **训练和测试**：完整 **pyproject.toml**、**uv.lock**；**tests/** 包含课程、动作、碰撞和 Vendi 等 CPU 单元测试。
- **预训练与生成资源**：官方 Hugging Face collection 链接 Stage 1 RL checkpoints、Stage 2 数据以及 Stage 3 学生权重；实际资源下载成功、大小和镜像速度仍待测。
- **仿真训练/评测**：有脚本及任务配置；但没有将所有论文基线（MP/MimicGen/PGDG/PLD/ExpertGen）的统一再训练流程全部打包到同样的一键入口。
- **真机**：有实验录像与论文协议，**但未确认存在可在陌生 FR3 平台上一键运行的完整真实部署、标定与安全接管流程**。

### 4.2 推荐的复现分层与命令

**第一层——无仿真验证（必须优先通过）**：环境依赖/CPU 单元测试、下载检查、读取 Stage 2 数据、离线碰撞阈值计算、Vendi 指标；应核对每一步的 sample/episode ID、动作维度和复现种子。

~~~bash
git clone https://github.com/DAVIAN-Robotics/DiscoDemo.git
cd DiscoDemo
uv sync
uv run pytest
~~~

**第二层——在受支持 RTX 节点验证作者权重**：先加载 Stage 3 公开 checkpoint 运行 **sft.sim_eval**，确认实际图像、动作尺度、控制延迟和成功率合理，再使用 Stage 1 生成器进行小规模数据导出；最后才考虑完整 50M-step RL 训练。

~~~bash
# 需 Isaac Sim 5.0/Isaac Lab 2.2 可用的 RTX 环境
python -m sft.sim_eval <STAGE3_CHECKPOINT> \
  --task pnp_banana --out-dir out/eval_pnp_banana

# RL generator（从零训练，默认规模大）
bash scripts/1_rl/train.sh pnp_banana discodemo
bash scripts/1_rl/train.sh pnp_banana prfcl
~~~

**第三层——论文级结果对齐**：对齐四任务环境、RL seed、3,000 条/任务、SFT 训练随机种子、50 个仿真评测 episode 及真机测试协议。原论文真机每方法每任务测试 20 次，只有有限随机种子，不能把报告均值当作跨平台鲁棒性保证。[D1]

### 4.3 核心风险登记

| 风险 | 证据 | 对复现的影响 | 处理策略 |
|---|---|---|---|
| **H100 不属于 Isaac Sim 官方支持的 RTX 图形卡** | 官方要求 RTX-capable GPU，Isaac Sim H100 不具 RT Cores | H100 可做 VLA/SFT 与离线计算，却无法直接保证 RTX 渲染/整套模拟器运行 | 先拿一张受支持 RTX 卡验证仿真；H100 用于学生训练 |
| 单 GPU 脚本不自动利用 8×H100 | Stage 1 和 Stage 3 官方脚本均以单个任务/进程运行 | 8 卡不是“一任务 8 倍速度” | 将卡分配给不同任务/seed；实测 wall-clock |
| 仿真安全统计与阈值表面矛盾 | 论文 Table 1 与 **collision_metrics.py** | 安全效益难以严格对齐 | 比较原始 states、keep.json、filter 前后同一集合 |
| 归档数据记录绝对配置路径 | **collect_rl_states.py** 与 **score_collision.py** | 跨机拷贝后可能找不到训练 YAML | 将配置快照一并归档，或以内容哈希/相对路径加载 |
| P-RFCL 的 z_dim 也变化 | **method/prfcl.yaml** z_dim=0，DiscoDemo z_dim=3 | alpha 消融同时改变了策略条件结构 | 补充 **z_dim=3, alpha=0** 控制组 |
| 真实 FR3 标定/安全 | 论文实验配置 + 没有独立一键真机部署 | 仿真成功不等于能复现 78% 真机成功率 | 如无设备，清楚标注仿真层级，不宣称完成真机 |

**算力判断**：论文主要工作量之一是 50M 环境步的仿真 RL；公开 README 的约 **15 GPU-hours/任务** 指生成器训练、成功轨迹采集与渲染量级，**不含 π0.5 SFT 与真机采集**；GPU 型号与开启的渲染功能不同会显著改变时间。8×H100×1–2 天具备一定模型训练预算，但因 Isaac Sim 图形支持限制，**不能仅凭 H100 数量保证整条链路可复现**。[D2]

**审计判级**：核心算法与下游仿真管线 **较完整（可执行性待测）**；真机复现 **依赖专用硬件与配置**；所有论文 baseline **并非均已一键开放**。

## 5. 贡献、局限性与研究延伸

**主要贡献**：

1. 重新定位 RL 的用途：不是最终控制器，而是**丰富成功解的数据生产器**，使“离线学生数据覆盖”成为显式研究对象。
2. 用并行逆向课程克服稀疏奖励启动困难；并行性降低训练墙钟时长，但不能将其误称为环境样本效率等比例提升。
3. 用成功门控 METRA 避免“仅为了多样而制造失败”，同时使用安全 shaping 与离线过滤控制不希望的接触。
4. 提供真实 FR3 多任务测试、覆盖度量、生成效率和修正性数据收集互补实验，使贡献不止停留在轨迹可视化。

**论文数据**：作者报告平均仿真成功率 **62% vs 40%**、真实成功率 **78% vs 39%**（最强对照）；主真机任务分别为香蕉放置 90%、方块堆叠 75%、圆孔插接 80%、异形插孔 65%。这些为论文报告值，不能直接视为独立重现。[D1]

**局限与边界**：

- 只验证四个固定 FR3 桌面任务；无法外推到移动操作、不同机器人本体或长时程多阶段任务。
- 依赖可重置状态、成功判定及接触传感器；真实接触模型不准时，安全过滤仍可能接受“仿真成功/真实失败”样本。
- **多样性最大化不等于学生收益最大化**；作者的 alpha 消融本身显示过强多样性会影响任务能力。
- 真机测试预算和训练随机种子有限；仿真/真机初态协议不能简单相减来计算严格 sim-to-real gap。
- 已发布静态数据的有效性与**生产这一数据所需的场景重建/真实设备工程**属于不同成本项。

**值得延伸的可验证问题**：

- **Useful Diversity**：同样的 3,000 条预算下，按 Coverage、student loss、不确定性、物理风险选择数据，能否比随机保留提升真实泛化？
- **Robust Skill Diversity**：在摩擦、物体质量、控制延迟和几何扰动变化时，仅奖励多个域下都成功的轨迹，是否比单域 METRA 更可迁移？
- **价值归因**：加入固定 z_dim=3、alpha=0 对照和等量 transition / 等时长训练；避免把表征容量、训练预算、轨迹长度混为多样性收益。

---

# 第二篇：DemoGen

**原题**：DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning  
**作者**：Zhengrong Xue、Shuying Deng、Zhenyang Chen、Yixuan Wang、Zhecheng Yuan、Huazhe Xu。  
**发表记录**：**Robotics: Science and Systems (RSS) 2025**；DOI: 10.15607/RSS.2025.XXI.157。  
**原文**：[RSS 正式论文页面](https://roboticsproceedings.org/rss21/p157.html)｜[arXiv](https://arxiv.org/abs/2502.16932)｜[HTML 全文](https://arxiv.org/html/2502.16932v1)｜[项目网站](https://demo-generation.github.io/)｜[GitHub 默认 master 分支](https://github.com/TEA-Lab/DemoGen/tree/master)

## 1. 摘要翻译（依据正式论文 Abstract 的中文译述）

视觉—动作策略已展示出很强的机器人操作能力，但要获得可靠表现，通常仍需要大量人工示教。其中一个关键原因是现有策略的**空间泛化能力有限**：当物体的位置与配置变化时，必须通过覆盖更多空间区域的演示来建立能力。

本文提出 DemoGen，一种成本较低、无需对每个新配置进行真实机器人重放的**全合成示教生成方法**。对每项任务仅需要一条人工采集的示教，方法便可根据新的物体初始配置调整已有动作轨迹，合成具有空间变化的示教数据。视觉观测采用三维点云，并通过对场景中对象和机械臂相关点云进行三维编辑来生成。

作者在多种真实机器人操作任务中验证了其有效性，包括柔性物体、灵巧手末端执行器和双臂平台。进一步地，该方法还可面向分布外场景扩展，生成带有外力扰动恢复和障碍回避行为的数据。[G1]

**一句话解读**：它没有像 DiscoDemo 那样训练 RL teacher 去发现新的操作解，而是**把已知技能中的可迁移片段整体变换，同时重新规划无接触的移动片段，并同步合成观测**。

## 2. 架构图/方法图 + 核心技术与算法

### 2.1 官方原图

![DemoGen 官方项目总体示意](https://raw.githubusercontent.com/TEA-Lab/DemoGen/master/pics/teaser.png)

*作者默认 master 分支的原始 teaser 图，展示真实示教扩展为不同物体配置的训练数据。*

![DemoGen 官方方法架构示意](https://raw.githubusercontent.com/TEA-Lab/DemoGen/master/pics/method.png)

*作者默认 master 分支的原始 method 图；图中涉及任务分段、动作生成与合成观测。详细描述见原论文 Section IV / Figures 4–6。*

### 2.2 中文方法架构

~~~mermaid
flowchart TD
  A["一条真实示教：点云、机器人状态、动作"] --> B["首帧对象分割：RGB-D→对象点云"]
  A --> C["按对象中心将时间轴解析为 Motion / Skill"]
  B --> D["采样新对象/目标位姿"]
  C --> E["Skill：沿新物体位姿整体做空间变换"]
  D --> E
  C --> F["Motion：重新规划连接相邻 Skill 的无接触轨迹"]
  E --> G["生成新的机器人动作与本体状态"]
  F --> G
  B --> H["对象点云和末端点云分离/变换/拼接"]
  D --> H
  G --> I["合成匹配的 3D 点云观测 + 动作标签"]
  H --> I
  I --> J["Zarr 训练集"]
  J --> K["DP3 或 2D Diffusion Policy 训练"]
  K --> L["真实单臂、灵巧手、双臂评测"]
~~~

### 2.3 运动分段：Motion 与 Skill

论文将任务拆成沿时间排列的 **motion**（自由空间移动）与 **skill**（含对象接触的操作）片段。例如：移动到花朵 → 抓花 → 把花移动到花瓶上方 → 插花。它们本质上对空间变换的允许程度不同：

- **Skill** 片段包含有接触约束的相对运动。若目标物体初始位姿发生变化，整个 skill 片段对齐新的对象坐标系，尽量保留示教中已经可靠的接触运动。
- **Motion** 片段负责连接相邻操作片段。源路径可能因对象位姿变化不再合适，因此在新的起止位姿之间使用插值或运动规划构造连接。
- **末端执行器/夹爪**：物体相关交互的 hand command（例如开闭夹爪、灵巧手关节）不随对象的平移任意修改。

设源对象初始变换为 T_old、目标变换为 T_new，则对齐变换可以写成（具体左右乘顺序依坐标定义）：

$$
\Delta T=T_{\mathrm{old}}^{-1}T_{\mathrm{new}}.
$$

对自由空间段，目标是把前一技能的终点与后一技能的起点连接起来，形式化为

$$
A_{\mathrm{motion}}=\mathrm{MotionPlan}(A_{\mathrm{skill,prev}}^{\mathrm{end}},A_{\mathrm{skill,next}}^{\mathrm{start}}).
$$

论文本质采用 TAMP 的“何时保留技能、何时规划路径”思想，但并没有发明一套覆盖所有接触物理的通用求解器。[G2]

### 2.4 观测合成：将点云对象视为可编辑场景元素

论文采用**点云作为策略的视觉表征**，区别于 RGB 图像生成型方法。

- 真实 RGB-D 相机捕获场景；裁掉背景/桌面，聚类去噪、使用 FPS（如 512/1024 点）下采样。
- 对源示教**首帧**分割对象，获得对象与目标点云；作者说明可人工框选，也提供 Grounded-SAM 的分割工具。
- 对新配置的对象点云施加空间变换；对机械臂末端的点云施加与新状态一致的变换。
- 将对象运动分为 **to-do**（尚未交互）、**doing**（与末端接触）、**done**（已完成），从而选择点云变换与拼接规则。
- 状态与动作标签同步变换，避免“新视觉—旧动作”配对失真。

这是 DemoGen 的核心效率来源：**不必为每条新演示启动完整动力学仿真或真机执行**。作者网站报告 22 秒生成 2,214 条轨迹、约 147,000 个 observation-action pairs；这是该生成设置的报告值，**不含下游扩散策略训练，也不等于任意任务从原始 RGB-D 开始的全部工程时间**。[G3]

### 2.5 扩展：ADR 与障碍回避

- **Augmentation for Disturbance Resistance (ADR)**：对对象与末端采用不同步的空间变换，合成“对象突然被移动→机器人重新接近→继续操作”的轨迹。作者在抹酱任务上报告 Normalized Score **92.3 vs 40.4**（常规 DemoGen），说明鲁棒性不是无条件由普通空间扰动自然涌现。
- **障碍回避**：在点云里加入障碍几何，再规划绕障路径，以合成带避障动作的示教。这里既引入了新观测，也改变了真实规划路径。
- 两种扩展应视为**特定任务数据生成模块**，不能默认为所有场景自动有效。[G4]

## 3. 完整 Pipeline：从一条真实示教到部署

| 阶段 | 输入 | 处理 | 公开实现或数据 | 主要风险 |
|---|---|---|---|---|
| 0 数据采集 | 实机相机、动作、本体状态 | 采集一条任务示教，点云/动作/状态对齐 | **real_world/collect_demo.py**；**data/datasets/source/*.zarr** | 机器人、相机同步；非全部硬件皆有适配 |
| 1 预处理 | RGB-D + Zarr | 空间裁剪、去噪、首帧对象分割 | **mask_util.py**、**data/sam_mask/** | 掩码错误直接污染新轨迹 |
| 2 分段 | 源时间序列 | 标注 motion/skill 交界帧或距离阈值自动解析 | **demogen.py** 的 parse_frames_* | 任务语义需人工调参 |
| 3 空间采样 | 新对象/目标位姿分布 | 网格/随机采样目标物体配置 | **demo_generation/config/*.yaml** | 越界、不可达、碰撞 |
| 4 合成动作 | 源技能 + 新配置 | 变换 skill、重算 motion、同步 state/hand | **demogen.py** | 非法接触或插值不连续 |
| 5 合成点云 | 首帧对象点云、原逐帧点云 | 运动阶段点云变换、对象与手的重组合 | **demogen.py** | 透视/遮挡/自遮挡造成 mismatch |
| 6 生成数据 | (state,action,point_cloud) | 写成新 Zarr dataset | **demo_generation/gen_demo.py** | 不自动保证生成轨迹真机成功 |
| 7 策略训练 | DemoGen Zarr | DP3/2D DP，固定训练预算及 seed | **diffusion_policies/train.sh** | 旧包兼容、学习率/步数对齐 |
| 8 真实评测 | 学生 policy | 真机在不同空间位置执行 | **real_world/evaluate.py** | Franka + Allegro / R1 的平台条件 |

**官方快速试跑**（在仓库根目录、按文档配置 Python 3.8 的环境后）：

~~~bash
git clone --branch master https://github.com/TEA-Lab/DemoGen.git
cd DemoGen

# 第一阶段建议只测试仓库自带的生成样例。
# 以下 shell 位于 demo_generation 目录内
cd demo_generation
bash run_gen_demo.sh

# 训练阶段（官方文档中的示例）
cd ../diffusion_policies
bash train.sh jar dp3 oyhand 0
~~~

作者的生成示例包括 Flower-Vase、Mug-Rack、Spatula-Egg、Sauce-Spreading；运行前需要依照 README 安装 **diffusion_policies** 与 **demo_generation** 的可编辑包。**run_gen_demo.sh 默认开启视频渲染**，而 README 说明视频渲染约每条 10 s，比实际轨迹合成贵得多；测吞吐时要关闭渲染，避免混淆开销。

## 4. 可复现性审查（GitHub 静态代码审计）

### 4.1 公开范围核实

- 仓库：**TEA-Lab/DemoGen**，**默认分支是 master 而不是 main**；本次检查树提交 **69a4931e8ef7694b5f61ebf3b7f393457500032f**。此前仅查询 main 会错误得出“只有 README/没有核心源码”的结论。
- **demo_generation/demo_generation/demogen.py**：核心类 DemoGen，包括一阶段/二阶段增强、点云变换、生成数据保存，核心方法确实公开。
- **demogen_bimanual.py**、**demogen_disturbance.py**、**demogen_obstacle.py**：多种机器人/目标需求的扩展实现存在，但接口与主入口统一程度不一致。
- **demo_generation/demo_generation/config/**：egg、flower、mug、sauce 任务配置；**data/datasets/source/** 包含实际 Zarr 示例；**data/sam_mask/** 包含分割相关内容。
- **diffusion_policies/**：DP3/DP 训练代码、配置、训练和评测入口；**real_world/** 有特定平台的采集与部署代码。
- **docs/0_install.md、1_data_collection.md、2_data_generation.md、3_train_policies.md**：安装、数据准备、算法使用和政策训练说明。

### 4.2 静态审计发现的具体复现障碍

**障碍 A：论文表达的通用 SE(3) 方法与 main 分支真实脚本有实现范围差。** 论文以 4×4 刚体变换说明对象中心增强；但公开 **demogen.py** 的常用 two_stage_augment() 分支集中实现 **xyz 平移、特定 motion/skill 次序与插值**，且函数注释包含“TODO：重构以支持任意对象数量和操作阶段”。因此不能据论文公式认定开源程序可以对任意复杂 6-DoF 姿态、任意对象数直接开箱运行。

**障碍 B：部分高级扩展仍存在作者机器路径。** **demogen_disturbance.py** 和 **demogen_obstacle.py** 中出现 **/home/dsy/shared/...**、**/home/zhengrong/...** 等路径；这意味着扩展模块需要重新配置数据路径，不能照搬脚本即运行。

**障碍 C：数据和任务预处理并非自动化零成本。** 需要对象分割掩码和时间分段，官方文档甚至推荐复杂任务人工指定分段帧。错误分割或接触边界会破坏生成点云、动作和物体的配对。

**障碍 D：老环境/新显卡兼容性。** 完整安装指引锁定 Python 3.8、torch 2.0.1、numba 0.56.4、hydra 1.2.0、zarr 2.12 等；它们与现代 CUDA/Python 环境可能冲突，不能直接保证在 H100 当前驱动上顺利安装。建议独立容器，优先用自带样例完成 smoke test。

**障碍 E：没有自动验证全部真机指标的 CI。** 可找到 train/eval 入口，但未找到覆盖全部真机任务与扩展功能的统一自动化测试；无同型号臂、相机、控制和初始化装置时，不能宣称论文级完整复现。

### 4.3 论文实验与可复制性层级

**仿真**：8 个改造 MetaWorld 任务中，论文 Table I 报告平均成功率：单源示教 **13%**、DemoGen **88%**、10 源示教 **68%**、25 源示教 **91%**（表内为各条件 **平均值**，不是最大值）。这是跨 3 个策略训练 seed 的仿真结果；可作为复现的目标范围，但必须使用作者修改过的空间随机化任务。[G5]

**真实**：8 项主任务共 **530 次 policy rollout**；作者 Table III 报告源示教训练平均 **11.0%**，DemoGen 平均 **74.6%**。覆盖 Panda 普通夹爪、Allegro 灵巧手与双臂机器人。**注意：真实实验虽称一条人工示教，但为获得多种视觉噪声，另将该示教重放两次，形成 3 条点云轨迹；每评测配置添加 9 个小尺度扰动合成样本**。严格比较人力成本时必须说明这层区别。[G5]

| 可复现目标 | 当前证据等级 | 最小要求 |
|---|---|---|
| 基础 DemoGen 点云/动作生成 | **高（静态）** | 安装环境、示例 Zarr、mask、配置 |
| DP3 基于生成数据训练 | **中高（静态）** | 旧依赖、GPU、配置和精确训练预算 |
| 完整 8 个真实任务实验 | **中低** | 真实同构设备、相机标定、评测物体 |
| ADR/障碍扩展 | **中低** | 路径改造、专用任务输入、额外测试 |
| 任意新机器人/新复杂任务 | **尚不具备通用保证** | 格式适配、对象语义/阶段分段、控制器验证 |

### 4.4 H100 训练预算与任务验收建议

DemoGen 的**轨迹生成**主要是几何计算和点云变换，不需要动用 8 张 H100；GPU 主要用于 **DP3/DP 训练**、分割特征模型等。H100 预算在模型训练方面较宽裕，但真正耗时可能是**环境兼容、任务自定义、分割和真实评测**，不应承诺整套论文在 1–2 天内复现。

建议先做 Flower-Vase：固定现有 source、统一掩码与解析帧，关闭视频渲染，验证动作/状态/点云 shape、坐标范围及有限样本可视化。随后分别训练 **1-source、DemoGen、10-source** 对照；如果没有真实机器人，先报告公开数据和仿真层级，不能将它冒充真机复现。

**审计判级**：主要数据增强算法和策略训练链路**较完整**；扩展模块和任意任务泛化程度**明显低于论文方法的抽象表达**。

## 5. 贡献、局限性与研究延伸

**主要贡献**：

1. 先证明视觉操作策略空间泛化高度依赖示教覆盖区域，再通过合成数据降低重复人工遥操作的成本。
2. TAMP 风格的“接触 skill 保留、自由 motion 重规划”，相比直接全轨迹平移更符合多阶段操作结构。
3. 点云对象编辑与动作同步变换，将视觉观测合成和轨迹生成结合，不需要按每个目标初态真实执行。
4. 覆盖普通夹爪、灵巧手、双臂、柔性物体，并明确探索了 ADR 与障碍回避扩展。
5. RSS 2025 正式发表且源码包含真实示例数据，复现门槛总体低于需要大型仿真训练的 RL 数据生成方法。

**限制与潜在偏差**：

- **单视角点云 mismatch**：移动物体后，点云仍保留源视角的可见表面，无法自动显露原来不可见的背面/遮挡。作者承认这一问题会导致生成数据收益饱和。
- 接触动作主要复用源技能，难以自动发现与原技能**拓扑完全不同**的抓取/操作路径；“配置多样性”并不等于“解决方式多样性”。
- 目标位置可达不代表真实机械臂必能执行；插值、IK、碰撞和对象操作约束仍需验证。
- 复杂长程、多接触、对象身份变化及非刚体行为没有通用开源适配；任务分段需要人为语义先验。
- 合成点云的几何正确性与学生策略真实环境的成功率属于不同评价维度。

**值得延伸的方向**：

- **接触保持的姿态/对象增强**：将当前较多偏平移的样例扩展到 SE(3) 对象位姿，同时加入可达性、碰撞和夹爪接触一致性检查。
- **Visibility-Aware Augmentation**：显式估计新视角下的遮挡、缺失点云和点密度变化，检测或修复不可信视觉合成。
- **Selective Data Synthesis**：不是密集采满网格，而是根据学生失败区和边际成功率提升决定下一个合成配置，比较固定生成预算下的数据收益。
- **Failure/Recovery Coverage**：将 ADR 扩展为更一般的偏离状态恢复轨迹，而非只有预定义对象位移模式。

---

# 横向比较：适合计算机专业研究者的复现与创新路径

## A. 技术差异一览

| 问题 | DiscoDemo | DemoGen |
|---|---|---|
| “多样”具体指什么？ | 多种成功 solution mode 与状态轨迹覆盖 | 多种对象初态、目标空间位置与点云空间配置 |
| 是否需要训练生成器？ | **需要**：FlashSAC/RFCL/METRA | **不需要**：几何轨迹编辑与运动规划 |
| 是否生成全新接触策略？ | 有可能通过 RL 学到不同成功策略 | 通常重用源接触技能，迁移其相对几何 |
| 新轨迹是否被动力学闭环执行？ | **是**：Isaac Sim 反馈和成功判定 | **否**：主生成步骤不是完整动力学 rollout |
| 如何处理物理可靠性？ | 任务成功、非法接触/插接/位移过滤 | 控制器/规划与技能结构先验，视觉/动力学局限仍在 |
| 下游学生 | DROID 预训练 π0.5，RGB + 本体感知 | DP3 / 2D DP，点云 + 本体感知 |
| 真机平台 | FR3 单臂 | Panda、灵巧手、R1 双臂等 |
| 对纯 H100 集群的适配 | SFT/分析友好，Isaac Sim RTX 渲染受限 | 数据合成/训练可分开，主要问题是旧环境 |
| 论文可复现性 | 生成器/任务/数据比较全；真机链路受限 | 生成算法/样例数据较全；复杂任务专用性较强 |

## B. 建议的实验序列（按风险由低到高）

1. **DemoGen 复现闭环**：Flower-Vase 原始示教 → 16/256 条网格增强 → DP3 → 评测；先验证轨迹/点云/动作的时序一致性。
2. **DiscoDemo 离线分析闭环**：下载公开 Stage 2 states/datasets → 复核 Vendi/coverage、样本数量、过滤前后指标 → 选择同预算数据进行 π0.5 微调。
3. **共性研究问题：数据多样性是否等于有效性？** 对相同 episode 数、相同动作 token/transition 数、相同训练步数分别比较 random、coverage、student-aware、safety-aware 筛选。
4. 有受支持 RTX 节点后，再进入 **DiscoDemo 生成器微调/重训**，研究 robust skill discovery 或跨仿真域下的多样性。
5. 有真机资源再做真实测试，严格记录对象分布、失败定义、相机配置、种子数与试验次数；未做真机不得宣称 zero-shot sim2real 增益。

## C. 本轮静态审计的验收边界

本文**已核实**原文实验协议、官方 Github 分支/目录、训练/生成/评测脚本、配置与主要算法控制流；**未执行**本地环境安装、GPU 训练、仿真轨迹生成、真实机器人测试或全套第三方基线复现。

尤其不应混淆三种不同的“可复现”：

- **方法级**：能生成新的轨迹/数据并通过局部正确性检查。
- **仿真实验级**：同环境、同数据、同评测设置下观察到与作者相近的效果。
- **论文完整级**：包括所有基线、不同随机种子、真实机器人与统计检验。

本次建议：**优先选 DemoGen 作为低门槛空间增强基线；将 DiscoDemo 定为多样化成功行为的强化学习主基线，在 RTX 条件满足时推进。** 两者的组合研究空间在于：用 DemoGen 高效率拓展初态，使用 DiscoDemo / 其他策略生成更多成功模式，再在相同训练预算下对有效覆盖和真实迁移收益进行归因。

---

## 文献、图源与代码链接（可核验）

**DiscoDemo**
- [D1] Park et al. (2026), [论文 HTML 原文及附录](https://arxiv.org/html/2610.32093v1)；[arXiv 记录](https://arxiv.org/abs/2610.32093)。
- [D2] [官方代码与 README](https://github.com/DAVIAN-Robotics/DiscoDemo)；[RFCL/RL 训练入口](https://github.com/DAVIAN-Robotics/DiscoDemo/blob/main/libs/FlashSAC/train_rfcl_robolab.py)；[METRA](https://github.com/DAVIAN-Robotics/DiscoDemo/blob/main/libs/FlashSAC/flash_rl/agents/utils/metra.py)。
- [D3] [碰撞与安全指标](https://github.com/DAVIAN-Robotics/DiscoDemo/blob/main/libs/FlashSAC/flash_rl/rfcl/collision_metrics.py)；[过滤命令](https://github.com/DAVIAN-Robotics/DiscoDemo/blob/main/libs/RoboLab/scripts/score_collision.py)。
- [D4] [论文 Section 4.4 / Table 1 与 Appendix E.3](https://arxiv.org/html/2610.32093v1#S4.S4)；当前主分支对应数据与阈值差异为本文审计疑点。
- [公开数据与权重集合](https://huggingface.co/collections/DAVIAN-Robotics/discodemo-6ac696e70d028fddf0c3dfeb)；[官方项目图及视频](https://davian-robotics.github.io/DiscoDemo/)。

**DemoGen**
- [G1] Xue et al. (2025), [RSS 正式会议论文记录](https://roboticsproceedings.org/rss21/p157.html)，[arXiv HTML 原文](https://arxiv.org/html/2502.16932v1)。
- [G2] [论文 Section IV：分段、TAMP 和点云生成](https://arxiv.org/html/2502.16932v1#S4)；[核心 demogen.py](https://github.com/TEA-Lab/DemoGen/blob/master/demo_generation/demo_generation/demogen.py)。
- [G3] [官方项目：示意图、演示视频与生成开销](https://demo-generation.github.io/)；[默认 master 分支](https://github.com/TEA-Lab/DemoGen/tree/master)。
- [G4] [论文 Section VII：ADR 与障碍回避](https://arxiv.org/html/2502.16932v1#S7)；[扩展源码目录](https://github.com/TEA-Lab/DemoGen/tree/master/demo_generation/demo_generation)。
- [G5] [论文 Table I / II / III 与真实评测协议](https://arxiv.org/html/2502.16932v1#S5)；[策略训练指南](https://github.com/TEA-Lab/DemoGen/blob/master/docs/3_train_policies.md)。

*图片归原论文作者及项目方所有；原始方法图通过 GitHub 官方仓库文件 URL 直接嵌入，未声称自行绘制。中文 Mermaid 图为本文依据论文与源码整理。*
