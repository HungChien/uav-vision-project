# UAV Vision Project

基于深度学习的无人机视觉感知与智能分析系统。

本仓库用于组织 DJI 实习任务中的数据理解、目标检测、目标跟踪、模型优化、部署导出和实验报告工作。当前阶段先建立项目目录骨架，后续逐步补充可复现代码和实验记录。

## Project Scope

- 数据集理解与 EDA：VisDrone、UAV123
- 目标检测：YOLO / Faster R-CNN / SSD 等模型训练与评估
- 目标跟踪：SORT / DeepSORT / Siamese 类方法实验
- 检测跟踪集成：视频流目标检测、多目标跟踪与结果可视化
- 模型优化：轻量化、剪枝、量化、ONNX / TensorRT 导出
- 项目交付：代码库、实验报告、模型权重、技术文档

## Repository Layout

```text
.
├── configs/              # 训练、评估、部署配置
├── data/                 # 本地数据集挂载位置，默认不提交大文件
├── docs/                 # 技术文档、周报、总结材料
├── models/               # 权重文件与导出模型，默认不提交大文件
├── notebooks/            # EDA 与实验型 notebook
├── outputs/              # 评估结果、可视化、日志、报告产物
├── scripts/              # 命令行入口脚本
├── src/uav_vision/       # 核心 Python 包
└── tests/                # 单元测试与轻量集成测试
```

## Status

Step 1: repository skeleton created.

