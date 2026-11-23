# Sim2Real Frontier Survey — 中文前沿技术路线综述

> **2025–2026** · 可执行场景生成 / 视觉真实性 / 物理可靠性 / 轨迹多样性 · 真机与仿真数据协同训练

本仓库系统整理 23 篇/组 Sim-to-Real 代表研究。每篇按八部分介绍：论文与机构、依据 Abstract 的中文译述、研究动机、作者主图、算法技术、中文完整流程、真实/仿真实验、适用边界。

## 阅读入口

- **[完整中文综述（推荐直接阅读）](docs/Sim2Real_前沿技术路线综述_2025-2026.md)**
- [第 2 章｜视觉真实性](docs/chapters/02-visual.md)：GSWorld、RE³SIM、RoboRender、TabletopGen、EmbodiedGen V2、SimFoundry、MANGO
- [第 3 章｜物理可靠性](docs/chapters/03-physics.md)：SPARR、PIN-WM
- [第 4 章｜轨迹多样性](docs/chapters/04-diversity.md)：RoboTwin 2.0、RoboDream、Real2Render2Real、SCIZOR、TUCO（交叉阅读 SimFoundry Digital Cousins）
- [第 5 章｜真实与仿真数据协同](docs/chapters/05-sim-real.md)：MetaGenSim、Sim-and-Real Co-Training
- [第 6 章｜具身学习的物理一致性](docs/chapters/06-physical-realism.md)：精选 7 篇，围绕可交互资产、Real-to-Sim 和动作条件世界模型
- [论文/代码/图源索引](papers/catalog.csv)｜[图源说明](figures/README.md)

## 原论文截图与流程图

目前已直接嵌入 **46 张来自原始论文 PDF 的截图（23 篇论文 × 2 张）**，每篇包含论文主图和方法架构图／第二张关键图；全部存储在 [assets/paper-screenshots](assets/paper-screenshots/)，在 GitHub 的 Markdown 页面上无需访问外部图床即可显示。

每张图标明作者原论文 Figure 编号、PDF 页码和原始链接。**SPARR** 使用 Figure 1 与 Figure 3，**Real2Render2Real** 的第二张是 Figure 2 数据效率与性能对照图（而非另一个网络架构图）。[查看逐篇图源索引](figures/README.md)；[查看原论文及截图裁切信息](assets/paper-screenshots/manifest.json)。

[中文自绘方法流程图](assets/methods/) 另有 14 张 SVG；它们与论文截图严格区分，正文仍保留 Mermaid Pipeline。

## 仓库结构

```text
README.md
docs/
  Sim2Real_前沿技术路线综述_2025-2026.md
  chapters/
    02-visual.md
    03-physics.md
    04-diversity.md
    05-sim-real.md
    06-physical-realism.md
assets/
  paper-screenshots/   # 46 张论文 PDF 截图 + 来源清单
  methods/             # 14 张自绘中文 SVG 方法流程图
  paper-figures/       # 旧版官网原图素材备份
figures/README.md      # 原图来源和显示方式
papers/catalog.csv     # 结构化文献索引（不使用论文等级）
scripts/publish_to_github.sh
scripts/capture_paper_figures.py
.github/workflows/capture-figures.yml
```

**§2.4「生成式可交互仿真环境构建」**，集中讨论从图像/文本/实景视频生成可执行仿真世界，并与 §4.6 的 Digital Cousins 多样性分析关联。

**本轮新增的重点**：§2.6 [MANGO](docs/chapters/02-visual.md#26-mango无配对图像翻译实现低成本跨视角视觉-sim-to-real)（轻量无配对视觉翻译）和 §4.7 [TUCO](docs/chapters/04-diversity.md#47-tuco面向-sim-to-real-协同训练的轨迹效用与集合覆盖优化)（目标域效用/覆盖驱动的模拟轨迹筛选）；均保留八部分模板、原论文图片、定量结果与明确的复现边界。

第 6 章聚焦可交互场景、真实数据约束和具身世界模型，不再收录偏材料/数值物理的研究；八部分方法分析模板保留。

本仓库独立研究 Sim-to-Real，不关联任何特定 instruct-WAM 课题。论文概述依据英文 Abstract 忠实中文译述；方法分析和横向比较为本调研独立整理。更新于 2026-11-23。
