"""
visualization.py
基于 Plotly 的可视化图表模块

包含：
- 设备利用率堆叠条形图
- 队列长度对比图
- 吞吐量时间序列 + 累计爬坡曲线
- 瓶颈热力图
- 方案对比雷达图 / 条形图
"""
from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import StationStats, ThroughputStats, SimulationResult
import analysis


# 工业控制台配色
COLOR_PALETTE = {
    "running": "#00CC96",
    "idle": "#636EFA",
    "blocked": "#EF553B",
    "breakdown": "#AB63FA",
    "background": "#0E1117",
    "grid": "#2A2D3E",
    "text": "#E6E6E6",
    "accent": "#FFA726",
}


def _base_layout(title: str, height: int = 450) -> Dict:
    """统一图表布局风格"""
    return dict(
        title=dict(text=title, font=dict(size=18, color=COLOR_PALETTE["text"])),
        paper_bgcolor=COLOR_PALETTE["background"],
        plot_bgcolor=COLOR_PALETTE["background"],
        font=dict(color=COLOR_PALETTE["text"]),
        height=height,
        margin=dict(l=60, r=40, t=60, b=40),
    )


def utilization_chart(station_stats: List[StationStats]) -> go.Figure:
    """设备利用率水平堆叠条形图：运行 / 空闲 / 阻塞 / 故障"""
    names = [s.name for s in station_stats]
    running = [s.utilization for s in station_stats]
    blocked = [s.blocked_ratio for s in station_stats]
    breakdown = [s.breakdown_ratio for s in station_stats]
    idle = [s.idle_ratio for s in station_stats]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="运行", y=names, x=running, orientation="h",
                         marker_color=COLOR_PALETTE["running"]))
    fig.add_trace(go.Bar(name="阻塞", y=names, x=blocked, orientation="h",
                         marker_color=COLOR_PALETTE["blocked"]))
    fig.add_trace(go.Bar(name="故障", y=names, x=breakdown, orientation="h",
                         marker_color=COLOR_PALETTE["breakdown"]))
    fig.add_trace(go.Bar(name="空闲", y=names, x=idle, orientation="h",
                         marker_color=COLOR_PALETTE["idle"]))

    fig.update_layout(
        **_base_layout("设备利用率构成（稳态）", height=420),
        barmode="stack",
        xaxis=dict(title="时间占比", gridcolor=COLOR_PALETTE["grid"],
                   tickformat=".0%", range=[0, 1]),
        yaxis=dict(title="工序", gridcolor=COLOR_PALETTE["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def queue_length_chart(station_stats: List[StationStats]) -> go.Figure:
    """平均队列长度与最大队列长度对比"""
    names = [s.name for s in station_stats]
    avg_q = [s.avg_queue_length for s in station_stats]
    max_q = [s.max_queue_length for s in station_stats]
    capacity = [s.buffer_capacity for s in station_stats]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="平均队列长度", x=names, y=avg_q,
                         marker_color=COLOR_PALETTE["accent"]))
    fig.add_trace(go.Scatter(name="最大队列长度", x=names, y=max_q,
                             mode="markers+lines",
                             marker=dict(size=10, color=COLOR_PALETTE["blocked"])))
    fig.add_trace(go.Scatter(name="缓冲区容量", x=names, y=capacity,
                             mode="lines", line=dict(dash="dash", color="#00BCD4")))

    fig.update_layout(
        **_base_layout("工序队列长度分析", height=420),
        xaxis=dict(title="工序", gridcolor=COLOR_PALETTE["grid"]),
        yaxis=dict(title="工件数", gridcolor=COLOR_PALETTE["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def queue_time_series_chart(station_stats: List[StationStats]) -> go.Figure:
    """各工序队列长度时间序列"""
    fig = go.Figure()
    for s in station_stats:
        if not s.queue_time_series:
            continue
        ts = np.array(s.queue_time_series)
        # 将分钟转换为仿真天，便于展示
        days = ts[:, 0] / (24 * 60)
        fig.add_trace(go.Scatter(
            x=days, y=ts[:, 1], mode="lines", name=s.name,
            line=dict(width=2),
        ))
    fig.update_layout(
        **_base_layout("队列长度动态变化", height=420),
        xaxis=dict(title="仿真天数", gridcolor=COLOR_PALETTE["grid"]),
        yaxis=dict(title="队列长度（件）", gridcolor=COLOR_PALETTE["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def throughput_chart(throughput: ThroughputStats, title: str = "产能爬坡与周吞吐量") -> go.Figure:
    """双 Y 轴：周吞吐量柱状图 + 累计产量面积图"""
    weeks = [f"第{i+1}周" for i in range(len(throughput.weekly_output))]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(name="周吞吐量", x=weeks, y=throughput.weekly_output,
               marker_color=COLOR_PALETTE["accent"]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(name="累计产量", x=weeks, y=throughput.cumulative_output,
                   mode="lines+markers", fill="tozeroy",
                   line=dict(color=COLOR_PALETTE["running"], width=3)),
        secondary_y=True,
    )

    fig.update_layout(
        **_base_layout(title, height=450),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="周吞吐量（件）", secondary_y=False,
                     gridcolor=COLOR_PALETTE["grid"])
    fig.update_yaxes(title_text="累计产量（件）", secondary_y=True,
                     gridcolor=COLOR_PALETTE["grid"])
    fig.update_xaxes(gridcolor=COLOR_PALETTE["grid"])
    return fig


def bottleneck_heatmap(station_stats: List[StationStats]) -> go.Figure:
    """瓶颈热力图：工序 × 瓶颈指标"""
    names = [s.name for s in station_stats]
    caps = np.array([s.theoretical_capacity for s in station_stats])
    utils = np.array([s.utilization for s in station_stats])
    waits = np.array([s.avg_wait_time for s in station_stats])
    blocks = np.array([s.blocked_probability for s in station_stats])

    # 归一化到 0-1，产能越低越严重 => 反向
    cap_norm = 1.0 - (caps / caps.max() if caps.max() else 1.0)
    util_norm = utils / utils.max() if utils.max() else 0.0
    wait_norm = waits / waits.max() if waits.max() else 0.0
    block_norm = blocks / blocks.max() if blocks.max() else 0.0

    z = np.column_stack([cap_norm, util_norm, wait_norm, block_norm])
    metrics = ["产能不足", "利用率", "等待时间", "阻塞概率"]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=metrics,
        y=names,
        colorscale=[[0, "#00CC96"], [0.5, "#FFA726"], [1, "#EF553B"]],
        showscale=True,
        zmin=0, zmax=1,
        text=np.round(z, 2),
        texttemplate="%{text}",
        hovertemplate="工序: %{y}<br>指标: %{x}<br>严重度: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("瓶颈热力图（红色越深越严重）", height=450),
        xaxis=dict(side="top", gridcolor=COLOR_PALETTE["grid"]),
        yaxis=dict(gridcolor=COLOR_PALETTE["grid"]),
    )
    return fig


def scenario_radar_chart(metrics: List[str], values: Dict[str, List[float]]) -> go.Figure:
    """方案对比雷达图"""
    fig = go.Figure()
    colors = ["#00CC96", "#EF553B", "#636EFA", "#AB63FA"]
    for i, (name, vals) in enumerate(values.items()):
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=metrics + [metrics[0]],
            fill="toself",
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.update_layout(
        **_base_layout("方案对比雷达图", height=520),
        polar=dict(
            bgcolor=COLOR_PALETTE["background"],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLOR_PALETTE["grid"]),
            angularaxis=dict(gridcolor=COLOR_PALETTE["grid"]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def scenario_bar_chart(results: List[SimulationResult]) -> go.Figure:
    """方案对比并排条形图：周吞吐量与达成率"""
    names = [r.config.mode_name or r.config.line_config.mode.value for r in results]
    tps = []
    achievements = []
    for r in results:
        cfg = r.config
        observation_weeks = max(1.0, cfg.sim_weeks - cfg.warmup_weeks)
        tp = r.throughput.total_output / observation_weeks
        tps.append(tp)
        achieve = (tp / cfg.line_config.target_weekly_output
                   if cfg.line_config.target_weekly_output else 0.0)
        achievements.append(achieve)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("周吞吐量", "目标达成率"),
                        horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=names, y=tps, marker_color=COLOR_PALETTE["accent"],
                         text=[f"{v:.0f}" for v in tps], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=achievements, marker_color=COLOR_PALETTE["running"],
                         text=[f"{v:.1%}" for v in achievements], textposition="outside"),
                  row=1, col=2)

    fig.update_layout(
        **_base_layout("方案关键指标对比", height=420),
        showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLOR_PALETTE["grid"])
    fig.update_xaxes(gridcolor=COLOR_PALETTE["grid"])
    fig.update_yaxes(tickformat=".0%", col=2)
    return fig


def station_metrics_table(station_stats: List[StationStats]) -> pd.DataFrame:
    """生成工序指标表格"""
    rows = []
    for s in station_stats:
        rows.append({
            "工序": s.name,
            "设备数": s.machine_count,
            "工人数": s.worker_count,
            "理论产能(件/周)": round(s.theoretical_capacity, 0),
            "利用率": f"{s.utilization:.1%}",
            "阻塞率": f"{s.blocked_probability:.1%}",
            "平均等待(min)": round(s.avg_wait_time, 1),
            "平均队长": round(s.avg_queue_length, 1),
            "最大队长": s.max_queue_length,
        })
    return pd.DataFrame(rows)
