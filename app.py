"""
app.py
基于 Streamlit 的汽车生产线优化平台 —— 现代化工业控制台界面

运行方式：
    streamlit run app.py

功能：
- 参数化配置 GA3 / GA3+GA4 产线
- 一键运行离散事件仿真
- 实时展示设备利用率、队列、吞吐量、瓶颈热力图
- GA3 vs GA3+GA4 方案对比
- 特斯拉 2018 产能地狱案例背景
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from models import SimulationConfig, LineMode, build_ga3_config, build_ga3_ga4_config
from simulation import run_simulation
from visualization import (
    utilization_chart,
    queue_length_chart,
    queue_time_series_chart,
    throughput_chart,
    bottleneck_heatmap,
    scenario_radar_chart,
    scenario_bar_chart,
    station_metrics_table,
)
import analysis


# ---------------------------------------------------------------------------
# 页面配置与样式
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="汽车生产线优化平台",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0E1117;
        --panel: #161B22;
        --accent: #FFA726;
        --text: #E6E6E6;
        --grid: #2A2D3E;
    }
    .main { background-color: var(--bg); color: var(--text); }
    .css-1d391kg, .css-1vq4p4l { background-color: var(--panel); }
    h1, h2, h3 { color: var(--accent) !important; letter-spacing: 0.5px; }
    .metric-card {
        background: linear-gradient(135deg, #1B1F27 0%, #232837 100%);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .metric-label { color: #9E9E9E; font-size: 0.85rem; }
    .metric-value { color: #FFFFFF; font-size: 1.6rem; font-weight: 700; }
    .stButton>button {
        background-color: var(--accent);
        color: #000000;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
    }
    .stButton>button:hover { background-color: #FFB74D; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏭 基于瓶颈理论与离散事件仿真的汽车生产线优化平台")
st.caption("以特斯拉 2018 年 Model 3 产能突破为案例背景 · SimPy + Plotly + Streamlit")


# ---------------------------------------------------------------------------
# 侧边栏：参数控制台
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ 生产参数控制台")

mode_label = st.sidebar.selectbox(
    "产线模式",
    options=[LineMode.GA3.value, LineMode.GA3_GA4.value],
    index=0,
)
mode = LineMode.GA3 if mode_label == LineMode.GA3.value else LineMode.GA3_GA4

line_count = st.sidebar.slider(
    "总装线数量（生产线数量）",
    min_value=1,
    max_value=3,
    value=1 if mode == LineMode.GA3 else 2,
    disabled=(mode == LineMode.GA3),
    help="GA3 模式固定为 1 条；GA3+GA4 模式可设置并行总装线数量",
)

workers = st.sidebar.slider(
    "总工人数",
    min_value=5,
    max_value=80,
    value=20 if mode == LineMode.GA3 else 40,
    step=1,
)

work_hours = st.sidebar.slider(
    "每周工作时长（小时）",
    min_value=40.0,
    max_value=168.0,
    value=100.0,
    step=4.0,
)

sim_weeks = st.sidebar.slider(
    "仿真时长（周）",
    min_value=1.0,
    max_value=12.0,
    value=4.0,
    step=0.5,
)

warmup_weeks = st.sidebar.slider(
    "预热期（周）",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.1,
)

seed = st.sidebar.number_input(
    "随机种子",
    min_value=0,
    max_value=9999,
    value=42,
    step=1,
)

# 设备数量配置
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 各工序设备数量")

equipment_counts: dict = {}
if mode == LineMode.GA3:
    default_ids = [("stamping", "冲压"), ("welding", "焊装"),
                   ("painting", "喷漆"), ("assembly", "总装"), ("testing", "检测")]
    defaults = {"stamping": 2, "welding": 3, "painting": 2, "assembly": 4, "testing": 2}
else:
    default_ids = [("stamping", "冲压"), ("welding", "焊装"), ("painting", "喷漆")]
    defaults = {"stamping": 3, "welding": 5, "painting": 4}

for pid, pname in default_ids:
    equipment_counts[pid] = st.sidebar.slider(
        f"{pname} ({pid})", min_value=1, max_value=12,
        value=defaults[pid], step=1,
    )

if mode == LineMode.GA3_GA4:
    assembly_machines = st.sidebar.slider(
        "单条总装线设备数", min_value=1, max_value=12, value=4, step=1
    )
    testing_machines = st.sidebar.slider(
        "单条检测线设备数", min_value=1, max_value=8, value=2, step=1
    )
    for idx in range(line_count):
        equipment_counts[f"assembly_L{idx+1}"] = assembly_machines
        equipment_counts[f"testing_L{idx+1}"] = testing_machines


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _build_line_config_cached(mode_value: str, equip_json: str, workers: int,
                              work_hours: float, line_count: int):
    """缓存产线配置构建（避免重复创建）"""
    import json
    equipment_counts = json.loads(equip_json)
    mode = LineMode.GA3 if mode_value == LineMode.GA3.value else LineMode.GA3_GA4
    if mode == LineMode.GA3:
        return build_ga3_config(machine_counts=equipment_counts,
                                workers=workers, work_hours=work_hours)
    else:
        return build_ga3_ga4_config(machine_counts=equipment_counts,
                                    workers=workers, work_hours=work_hours,
                                    assembly_lines=line_count)


def build_line_config():
    import json
    return _build_line_config_cached(
        mode.value, json.dumps(equipment_counts), workers, work_hours, line_count
    )


def run_current_simulation(mode_name: str):
    """运行当前参数下的仿真"""
    line_cfg = build_line_config()
    sim_cfg = SimulationConfig(
        line_config=line_cfg,
        total_workers=workers,
        work_hours_per_week=work_hours,
        sim_weeks=sim_weeks,
        warmup_weeks=warmup_weeks,
        seed=int(seed),
        mode_name=mode_name,
    )
    result = run_simulation(sim_cfg)
    return result


def run_named_simulation(mode: LineMode, line_count: int, mode_name: str):
    """按指定模式运行仿真，用于方案对比"""
    if mode == LineMode.GA3:
        line_cfg = build_ga3_config(machine_counts=equipment_counts,
                                    workers=workers, work_hours=work_hours)
    else:
        line_cfg = build_ga3_ga4_config(machine_counts=equipment_counts,
                                        workers=workers, work_hours=work_hours,
                                        assembly_lines=line_count)
    sim_cfg = SimulationConfig(
        line_config=line_cfg,
        total_workers=workers,
        work_hours_per_week=work_hours,
        sim_weeks=sim_weeks,
        warmup_weeks=warmup_weeks,
        seed=int(seed),
        mode_name=mode_name,
    )
    return run_simulation(sim_cfg)


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 主页面：Tab 导航
# ---------------------------------------------------------------------------
tab_sim, tab_bottleneck, tab_compare, tab_case = st.tabs([
    "🎛️ 仿真控制台", "🔍 瓶颈分析", "📊 方案对比", "📖 案例背景"
])

# ---------------------------------------------------------------------------
# Tab 1: 仿真控制台
# ---------------------------------------------------------------------------
with tab_sim:
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("▶️ 运行仿真", key="run_sim")

    if run_btn:
        with st.spinner("正在运行离散事件仿真，请稍候..."):
            result = run_current_simulation(mode_name=mode.value)
            st.session_state["result"] = result

    result = st.session_state.get("result")
    if result is None:
        st.info("请点击左侧「运行仿真」按钮开始仿真。")
    else:
        cfg = result.config
        throughput_weekly = result.throughput.total_output / max(1.0, cfg.sim_weeks - cfg.warmup_weeks)
        achievement = (throughput_weekly / cfg.line_config.target_weekly_output
                       if cfg.line_config.target_weekly_output else 0.0)
        avg_util = np.mean([s.utilization for s in result.station_stats])
        oee = avg_util * 0.95 * 0.98
        wip = analysis._estimate_wip(result.station_stats)

        # 指标卡片区
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            metric_card("周吞吐量", f"{throughput_weekly:.0f} 件/周")
        with c2:
            metric_card("目标达成率", f"{achievement:.1%}")
        with c3:
            metric_card("第一瓶颈", result.bottleneck.theoretical_bottleneck_name)
        with c4:
            metric_card("平均设备利用率", f"{avg_util:.1%}")
        with c5:
            metric_card("OEE", f"{oee:.1%}")
        with c6:
            metric_card("平均在制品 WIP", f"{wip:.1f}")

        st.markdown("---")

        # 图表区
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(utilization_chart(result.station_stats),
                            use_container_width=True, key="util_chart")
        with g2:
            st.plotly_chart(queue_length_chart(result.station_stats),
                            use_container_width=True, key="queue_chart")

        g3, g4 = st.columns(2)
        with g3:
            st.plotly_chart(throughput_chart(result.throughput),
                            use_container_width=True, key="tp_chart")
        with g4:
            st.plotly_chart(bottleneck_heatmap(result.station_stats),
                            use_container_width=True, key="heat_chart")

        st.markdown("#### 队列长度动态变化")
        st.plotly_chart(queue_time_series_chart(result.station_stats),
                        use_container_width=True, key="queue_ts_chart")

        st.markdown("#### 工序统计明细")
        st.dataframe(station_metrics_table(result.station_stats),
                     use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: 瓶颈分析
# ---------------------------------------------------------------------------
with tab_bottleneck:
    result = st.session_state.get("result")
    if result is None:
        st.info("请先在「仿真控制台」运行仿真。")
    else:
        b = result.bottleneck
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("理论瓶颈", b.theoretical_bottleneck_name)
        with c2:
            st.metric("利用率瓶颈", b.utilization_bottleneck_name)
        with c3:
            st.metric("排队瓶颈", b.queue_bottleneck_name)
        with c4:
            st.metric("DBR 时间缓冲", f"{b.time_buffer:.1f} min")

        st.markdown("#### 综合瓶颈排序")
        rank_df = pd.DataFrame(b.ranked_stations)
        display_df = rank_df[["name", "overall_score", "capacity", "utilization",
                              "avg_wait_time", "avg_queue_length", "queue_health_index", "suggestion"]]
        display_df = display_df.rename(columns={
            "name": "工序", "overall_score": "瓶颈评分",
            "capacity": "理论产能(件/周)", "utilization": "利用率",
            "avg_wait_time": "平均等待(min)", "avg_queue_length": "平均队长",
            "queue_health_index": "排队健康度", "suggestion": "优化建议",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("#### TOC 五步聚焦法")
        st.markdown(
            """
            1. **识别约束**：遍历所有工序，理论产能最小者即为当前瓶颈。
            2. **充分利用约束**：保证瓶颈工序不停机待料，实施质量前置控制。
            3. **其他资源从属**：非瓶颈工序按瓶颈节拍投料，避免过量在制品。
            4. **提升约束能力**：增加瓶颈设备、缩短换模时间、提升可用率。
            5. **防止惰性**：瓶颈缓解后返回步骤 1，持续识别新瓶颈。
            """
        )


# ---------------------------------------------------------------------------
# Tab 3: 方案对比
# ---------------------------------------------------------------------------
with tab_compare:
    st.markdown("#### GA3 单线 vs GA3+GA4 双线")
    st.caption("使用相同随机种子与工时参数，公平对比两种产线组织模式。")

    compare_btn = st.button("▶️ 运行方案对比", key="run_compare")

    if compare_btn:
        with st.spinner("正在顺序运行两种方案仿真..."):
            result_ga3 = run_named_simulation(LineMode.GA3, 1, "GA3 单线")
            result_ga4 = run_named_simulation(LineMode.GA3_GA4, 2, "GA3 + GA4 双线")
            st.session_state["compare_results"] = [result_ga3, result_ga4]

    cmp_results = st.session_state.get("compare_results")
    if cmp_results:
        st.markdown("##### 对比汇总表")
        st.dataframe(analysis.build_comparison_table(cmp_results),
                     use_container_width=True, hide_index=True)

        metrics, values = analysis.scenario_radar_data(cmp_results)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(scenario_radar_chart(metrics, values),
                            use_container_width=True, key="radar_chart")
        with c2:
            st.plotly_chart(scenario_bar_chart(cmp_results),
                            use_container_width=True, key="bar_cmp_chart")

        st.markdown("##### 指标变化明细")
        st.dataframe(analysis.compare_two_scenarios(cmp_results[0], cmp_results[1]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("点击「运行方案对比」生成 GA3 / GA3+GA4 对比结果。")


# ---------------------------------------------------------------------------
# Tab 4: 案例背景
# ---------------------------------------------------------------------------
with tab_case:
    st.markdown('### 特斯拉 2018 "生产地狱" 案例')
    st.markdown(
        """
        **背景**：2017 年 7 月 Model 3 正式交付，马斯克提出年底周产 5000 辆的激进目标。
        然而 2017 年 Q3 仅交付 222 辆，Q4 交付 1542 辆，产能爬坡严重滞后。
        2018 年初，媒体将弗里蒙特工厂的困境称为 **"生产地狱"（Production Hell）**。

        **关键事件时间线**：
        - **2017 Q4**：电池模组装配成为首要瓶颈，PACK 节拍远低于车身节拍。
        - **2018 Q1-Q2**：GA3 总装线受限，自动化焊接机器人可用率不足。
        - **2018 Q2 末**：喷漆车间因多色切换时间过长成为新瓶颈。
        - **2018 年 6 月**：在工厂外临时帐篷中搭建 GA4 产线，作为产能溢出通道。
        - **2018 年 6 月底**：最后一周实现 5031 辆 Model 3 下线，达成周产 5000 目标。

        **运筹学映射**：
        | 案例现象 | 运筹学概念 |
        |---|---|
        | 瓶颈在不同工序间漂移 | 动态约束 / TOC 五步聚焦法 |
        | 工件在工位前排队 | G/G/c 排队网络 |
        | 下游满导致上游停机 | 阻塞（Blocking） |
        | 增加 GA4 帐篷线 | 容量扩张 / 资源配置 |
        | 马斯克驻守 GA3 | 管理关注瓶颈（DBR） |

        **本平台价值**：通过离散事件仿真复现上述动态过程，帮助学生与决策者
        直观理解瓶颈理论、排队论与产线平衡在真实制造系统中的作用。
        """
    )

    st.markdown("### 教学讨论题")
    st.markdown(
        """
        1. 为什么增加一条总装线（GA4）不能使产能简单翻倍？
        2. 当瓶颈从焊装漂移到喷漆时，应该优先调整哪些参数？
        3. 设备利用率越高越好吗？高利用率可能带来什么问题？
        4. 如何在本平台中验证 TOC 的"约束决定系统产出"论断？
        """
    )
