# DiscoDemo：面向 Sim2Real 的轨迹多样性生成——方法、实验及可复现性深度调研

> **更新：2026-11-23** ｜**主题：轨迹多样性、强化学习、模仿学习、Sim2Real**｜**依据：论文原文 + 官方代码 + 静态审计**  
> 本文是独立研究笔记，不替换仓库中的[完整中文综述](Sim2Real_前沿技术路线综述_2025-2026.md)或[轨迹多样性章节](chapters/04-diversity.md)。  
> **说明**：摘要为忠实中文译述；性能数据均为作者报告，并非本研究运行所得。源码存在不等于已通过端到端复现，文中明确标示静态审计疑点。

## 研究定位与结论

**DiscoDemo（2026 年 10 月 arXiv 预印本）**研究如何从少量真实机器人示教出发，在真实场景对应的 Isaac Sim 仿真环境中使用带成功门控的多技能强化学习，生成多条**不同且成功**的机器人操作轨迹，再筛选并渲染为视觉模仿学习数据，以提高无额外真实微调的真机任务成功率。

- **研究价值**：主动发现不同成功解，而不只是复制或空间扰动原轨迹；真实 FR3 四任务报告了显著提升。
- **复现价值**：RL 生成器、课程、METRA、数据过滤、种子示教、策略训练与仿真评测入口公开，并有部分预训练权重与数据。
- **现实障碍**：Isaac Sim 5.0/Isaac Lab 2.2 的 RTX 环境要求、真实 FR3 部署标定，以及论文安全统计与当前过滤阈值的待解释差异。
- **推荐路径**：先复核公开数据、安全指标和 Vendi 覆盖度，再验证已发布 π0.5 仿真结果；具备 RTX 节点后训练 RL 生成器，最后开展真机实验。

---

## 论文基本信息

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

## 参考文献与代码出处

- [D1] Park et al. (2026), [论文 HTML 原文及附录](https://arxiv.org/html/2610.32093v1)；[arXiv 记录](https://arxiv.org/abs/2610.32093)。
- [D2] [官方代码与 README](https://github.com/DAVIAN-Robotics/DiscoDemo)；[RFCL/RL 训练入口](https://github.com/DAVIAN-Robotics/DiscoDemo/blob/main/libs/FlashSAC/train_rfcl_robolab.py)；[METRA](https://github.com/DAVIAN-Robotics/DiscoDemo/blob/main/libs/FlashSAC/flash_rl/agents/utils/metra.py)。
- [D3] [碰撞与安全指标](https://github.com/DAVIAN-Robotics/DiscoDemo/blob/main/libs/FlashSAC/flash_rl/rfcl/collision_metrics.py)；[过滤命令](https://github.com/DAVIAN-Robotics/DiscoDemo/blob/main/libs/RoboLab/scripts/score_collision.py)。
- [D4] [论文 Section 4.4 / Table 1 与 Appendix E.3](https://arxiv.org/html/2610.32093v1#S4.S4)；当前主分支对应数据与阈值差异为本文审计疑点。
- [公开数据与权重集合](https://huggingface.co/collections/DAVIAN-Robotics/discodemo-6ac696e70d028fddf0c3dfeb)；[官方项目图及视频](https://davian-robotics.github.io/DiscoDemo/)。


*作者图片版权归原项目方；中文 Mermaid 图为本文依据论文和代码整理。此文尚未执行 GPU 训练或真机测试。*
