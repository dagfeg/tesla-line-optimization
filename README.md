# 基于瓶颈理论与离散事件仿真的汽车生产线优化平台

以特斯拉 2018 年 Model 3 “生产地狱”产能突破为案例背景，结合 **TOC 瓶颈理论**、**排队论**与 **SimPy 离散事件仿真**，构建的可交互式教学与决策支持平台。

## 技术栈

- Python 3.10+
- Streamlit（前端控制台）
- SimPy（离散事件仿真引擎）
- Plotly（可视化）
- Pandas / NumPy（数据分析）

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

或如果 `streamlit` 命令未加入 PATH：

```bash
python -m streamlit run app.py
```

## 核心功能

- **五大工艺段仿真**：冲压、焊装、喷漆、总装、检测
- **两种产线模式**：GA3 单线模式、GA3 + GA4 双线模式
- **自动瓶颈识别**：理论瓶颈、利用率瓶颈、排队瓶颈与综合瓶颈评分
- **参数化调整**：工人数、设备数、工作时长、总装线数量
- **Plotly 可视化**：设备利用率图、队列长度图、吞吐量图、瓶颈热力图、方案对比雷达图
- **案例教学模块**：特斯拉 2018 年产能爬坡案例背景与讨论题

## 部署到 Streamlit Community Cloud

1. 在 GitHub 上新建一个公开仓库（例如 `tesla-line-optimization`）。
2. 将本项目代码推送到该仓库：

```bash
git init
git add .
git commit -m "init"
git branch -M main
git remote add origin https://github.com/你的用户名/tesla-line-optimization.git
git push -u origin main
```

3. 打开 [share.streamlit.io](https://share.streamlit.io)，使用 GitHub 账号登录。
4. 点击 **New app**，选择刚才推送的仓库，主文件路径填写 `app.py`，点击 **Deploy**。
5. 部署完成后会获得一个永久公网链接，例如：

```
https://tesla-line-optimization-你的用户名.streamlit.app
```

## 项目结构

```
.
├── app.py              # Streamlit 主界面
├── simulation.py       # SimPy 离散事件仿真引擎
├── models.py           # 数据模型与默认产线配置
├── analysis.py         # 瓶颈识别与方案对比
├── visualization.py    # Plotly 可视化
├── requirements.txt    # Python 依赖
├── .streamlit/
│   └── config.toml     # 暗色工业控制台主题
└── README.md
```

## 许可

本项目为运筹学课程设计/教学演示用途。
