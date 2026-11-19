# 作者原始论文截图索引（2025–2026）

中文综述每篇直接插入两张**从原论文 PDF 截取的图**；文件保存在仓库内而非外部图床。下方页码表示本次采用的 PDF 页码（从 1 开始），可能与论文印刷页码不同。

| 论文 | 论文主图（Figure 及 PDF 页码） | 方法架构图 / 第二张关键图 | 原论文 PDF |
|---|---|---|---|
| GSWorld | [Fig.1 · p.1](../assets/paper-screenshots/GSWorld-overview.jpg) | [Fig.2 · p.3](../assets/paper-screenshots/GSWorld-architecture.jpg) | [论文原文](https://3dgsworld.github.io/resources/gsworld.pdf) |
| RE³SIM | [Fig.1 · p.1](../assets/paper-screenshots/RE3SIM-overview.jpg) | [Fig.2 · p.3](../assets/paper-screenshots/RE3SIM-architecture.jpg) | [论文原文](https://arxiv.org/pdf/2502.08645) |
| RoboRender | [Fig.1 · p.1](../assets/paper-screenshots/RoboRender-overview.jpg) | [Fig.2 · p.3](../assets/paper-screenshots/RoboRender-architecture.jpg) | [论文原文](https://arxiv.org/pdf/2610.09254) |
| TabletopGen | [Fig.1 · p.2](../assets/paper-screenshots/TabletopGen-overview.jpg) | [Fig.2 · p.6](../assets/paper-screenshots/TabletopGen-architecture.jpg) | [论文原文](https://arxiv.org/pdf/2512.01204) |
| EmbodiedGen V2 | [Fig.1 · p.1](../assets/paper-screenshots/EmbodiedGenV2-overview.jpg) | [Fig.2 · p.4（资产生成）](../assets/paper-screenshots/EmbodiedGenV2-architecture.jpg) | [论文原文](https://arxiv.org/pdf/2607.07459) |
| SimFoundry | [Fig.1 · p.1](../assets/paper-screenshots/SimFoundry-overview.jpg) | [Fig.2 · p.4](../assets/paper-screenshots/SimFoundry-architecture.jpg) | [论文原文](https://arxiv.org/pdf/2606.28276) |
| SPARR | [Fig.1 · p.1](../assets/paper-screenshots/SPARR-overview.jpg) | [Fig.3 · p.4](../assets/paper-screenshots/SPARR-architecture.jpg) | [论文原文](https://arxiv.org/pdf/2602.23253) |
| SoMA | [Fig.1 · p.1](../assets/paper-screenshots/SoMA-overview.jpg) | [Fig.2 · p.3](../assets/paper-screenshots/SoMA-architecture.jpg) | [论文原文](https://raw.githubusercontent.com/mlresearch/v306/main/assets/huang26bh/huang26bh.pdf) |
| PIN-WM | [Fig.1 · p.1](../assets/paper-screenshots/PIN-WM-overview.jpg) | [Fig.2 · p.4](../assets/paper-screenshots/PIN-WM-architecture.jpg) | [论文原文](https://www.roboticsproceedings.org/rss21/p153.pdf) |
| RoboTwin 2.0 | [Fig.1 · p.2](../assets/paper-screenshots/RoboTwin-overview.jpg) | [Fig.2 · p.4](../assets/paper-screenshots/RoboTwin-architecture.jpg) | [论文原文](https://raw.githubusercontent.com/mlresearch/v306/main/assets/chen26j/chen26j.pdf) |
| RoboDream | [Fig.1 · p.1](../assets/paper-screenshots/RoboDream-overview.jpg) | [Fig.2 · p.3](../assets/paper-screenshots/RoboDream-architecture.jpg) | [论文原文](https://arxiv.org/pdf/2606.02577) |
| Real2Render2Real | [Fig.1 · p.2](../assets/paper-screenshots/R2R2R-overview.jpg) | [Fig.2 · p.3（实验图）](../assets/paper-screenshots/R2R2R-architecture.jpg) | [论文原文](https://raw.githubusercontent.com/mlresearch/v305/main/assets/yu25a/yu25a.pdf) |
| SCIZOR | [Fig.1 · p.2](../assets/paper-screenshots/SCIZOR-overview.jpg) | [Fig.2 · p.4](../assets/paper-screenshots/SCIZOR-architecture.jpg) | [论文原文](https://arxiv.org/pdf/2505.22626) |
| MetaGenSim | [Fig.1 · p.1](../assets/paper-screenshots/MetaGenSim-overview.jpg) | [Fig.2 · p.4](../assets/paper-screenshots/MetaGenSim-architecture.jpg) | [论文原文](https://arxiv.org/pdf/2610.19479) |
| Sim-and-Real Co-Training | [Fig.1 · p.1](../assets/paper-screenshots/CoTraining-overview.jpg) | [Fig.2 · p.4](../assets/paper-screenshots/CoTraining-architecture.jpg) | [论文原文](https://www.roboticsproceedings.org/rss21/p109.pdf) |

## 图像来源与使用说明

1. 截图源于作者原始 PDF 中对应 Figure；只截取、裁切和编码保存，不使用图像生成模型重新绘制。请引用作者论文。
2. [截图溯源 Manifest](../assets/paper-screenshots/manifest.json) 包含来源 PDF、Figure 编号、PDF 页码、裁切区域及文件路径。
3. [Python 提取脚本](../scripts/capture_paper_figures.py) 与 [GitHub Actions 工作流](../.github/workflows/capture-figures.yml) 可用于重新提取和修复裁切。
4. 自绘的中文流程图保存在 [assets/methods](../assets/methods/)，不同于原论文截图；另外一些历史作者网站图片仍在 `assets/paper-figures/`。
5. **新加入的三篇论文**：TabletopGen Figure 1/2、EmbodiedGen V2 Figure 1/2、SimFoundry Figure 1/2 均依据 2026-11-19 上传的相应 PDF 选择并核对；EmbodiedGen V2 的 Figure 2 只覆盖其资产生成分支，不能当成全部世界生成系统的唯一架构图。
6. **Real2Render2Real** 的 Figure 1 已展示数据生产全流程，Figure 2 属于数据生成效率与真实策略表现，未将其误称为“架构图”。

图片著作权归各论文的权利人所有；本仓库仅用于带出处的学术调研阅读。
