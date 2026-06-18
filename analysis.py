"""
analysis.py
瓶颈识别、排队指标与方案对比分析模块

核心功能：
- TOC 五步聚焦法算法化：自动识别产能瓶颈、利用率瓶颈、排队瓶颈
- 综合瓶颈评分（加权 TOPSIS 思路）
- 排队健康度指数 QHI
- 方案对比矩阵与 ROI 估算
"""
from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from models import (
    SimulationConfig,
    SimulationResult,
    StationStats,
    ThroughputStats,
    BottleneckReport,
)


# ---------------------------------------------------------------------------
# TOC / 瓶颈识别
# ---------------------------------------------------------------------------
def analyze_bottleneck(station_stats: List[StationStats],
                       throughput: ThroughputStats,
                       config: SimulationConfig) -> BottleneckReport:
    """根据工序统计结果生成瓶颈分析报告"""
    if not station_stats:
        return BottleneckReport(
            theoretical_bottleneck_id="",
            theoretical_bottleneck_name="",
            capacity_bottleneck_id="",
            utilization_bottleneck_id="",
            utilization_bottleneck_name="",
            queue_bottleneck_id="",
            queue_bottleneck_name="",
            ranked_stations=[],
            drum_rate=0.0,
            time_buffer=0.0,
        )

    # 1. 按理论产能排序（产能最小为瓶颈）
    by_capacity = sorted(station_stats, key=lambda s: s.theoretical_capacity)
    cap_bottleneck = by_capacity[0]

    # 2. 按利用率排序
    by_util = sorted(station_stats, key=lambda s: s.utilization, reverse=True)
    util_bottleneck = by_util[0]

    # 3. 按平均等待时间排序
    by_wait = sorted(station_stats, key=lambda s: s.avg_wait_time, reverse=True)
    queue_bottleneck = by_wait[0]

    # 4. 综合评分
    ranked = _rank_stations(station_stats)

    # DBR 参数
    observation_weeks = max(1.0, config.sim_weeks - config.warmup_weeks)
    actual_throughput = throughput.total_output / observation_weeks
    drum_rate = actual_throughput / (config.work_hours_per_week * 60.0) if actual_throughput else 0.0
    # 时间缓冲建议：瓶颈前等待时间的 1.5 倍
    time_buffer = cap_bottleneck.avg_wait_time * 1.5

    return BottleneckReport(
        theoretical_bottleneck_id=cap_bottleneck.process_id,
        theoretical_bottleneck_name=cap_bottleneck.name,
        capacity_bottleneck_id=cap_bottleneck.process_id,
        utilization_bottleneck_id=util_bottleneck.process_id,
        utilization_bottleneck_name=util_bottleneck.name,
        queue_bottleneck_id=queue_bottleneck.process_id,
        queue_bottleneck_name=queue_bottleneck.name,
        ranked_stations=ranked,
        drum_rate=drum_rate,
        time_buffer=time_buffer,
    )


def _rank_stations(station_stats: List[StationStats]) -> List[Dict]:
    """综合瓶颈评分排序，返回结构化列表"""
    caps = np.array([s.theoretical_capacity for s in station_stats])
    utils = np.array([s.utilization for s in station_stats])
    waits = np.array([s.avg_wait_time for s in station_stats])
    blocks = np.array([s.blocked_probability for s in station_stats])

    max_cap = caps.max() if caps.size else 1.0
    max_util = utils.max() if utils.size else 1.0
    max_wait = waits.max() if waits.max() > 0 else 1.0
    max_block = blocks.max() if blocks.max() > 0 else 1.0

    rows = []
    for s in station_stats:
        capacity_score = 1.0 - (s.theoretical_capacity / max_cap)
        util_score = s.utilization / max_util if max_util else 0.0
        wait_score = min(1.0, s.avg_wait_time / max_wait) if max_wait else 0.0
        block_score = s.blocked_probability / max_block if max_block else 0.0

        overall = (0.40 * capacity_score +
                   0.25 * util_score +
                   0.20 * wait_score +
                   0.15 * block_score)
        qhi = _queue_health_index(s)

        suggestion = _suggest(s, capacity_score, util_score, wait_score, block_score)
        rows.append({
            "process_id": s.process_id,
            "name": s.name,
            "overall_score": float(overall),
            "capacity": s.theoretical_capacity,
            "utilization": s.utilization,
            "avg_wait_time": s.avg_wait_time,
            "blocked_probability": s.blocked_probability,
            "avg_queue_length": s.avg_queue_length,
            "queue_health_index": qhi,
            "suggestion": suggestion,
        })

    rows.sort(key=lambda r: r["overall_score"], reverse=True)
    return rows


def _queue_health_index(s: StationStats) -> float:
    """排队健康度指数：0-100，越高越健康"""
    # 利用率、等待时间、阻塞概率、队长均做反向惩罚
    util_penalty = min(1.0, s.utilization)
    wait_penalty = min(1.0, s.avg_wait_time / 120.0)
    block_penalty = min(1.0, s.blocked_probability / 0.30)
    queue_penalty = min(1.0, s.avg_queue_length / s.buffer_capacity) if s.buffer_capacity else 0.0
    penalty = (0.30 * util_penalty +
               0.30 * wait_penalty +
               0.25 * block_penalty +
               0.15 * queue_penalty)
    return max(0.0, 100.0 - 100.0 * penalty)


def _suggest(s: StationStats, cap_score: float, util_score: float,
             wait_score: float, block_score: float) -> str:
    """根据瓶颈维度生成管理建议"""
    scores = {
        "产能不足": cap_score,
        "利用率过高": util_score,
        "等待过长": wait_score,
        "阻塞严重": block_score,
    }
    top = max(scores, key=scores.get)
    if top == "产能不足":
        return f"建议增加 {s.name} 并行设备或提升设备可用率，当前理论产能 {s.theoretical_capacity:.0f} 件/周"
    if top == "利用率过高":
        return f"建议检查 {s.name} 设备故障与换模时间，适当补充人力或预防性维护"
    if top == "等待过长":
        return f"建议在 {s.name} 前扩大缓冲区或提升上游供料稳定性"
    return f"建议增大 {s.name} 下游缓冲容量或提升下游产能以缓解阻塞"


# ---------------------------------------------------------------------------
# 排队论近似（Allen-Cunneen）
# ---------------------------------------------------------------------------
def allen_cunneen_wait(arrival_rate: float, service_rate: float, c: int,
                       cv_arrival: float, cv_service: float) -> float:
    """
    Allen-Cunneen 近似公式估算 G/G/c 平均等待时间：
    Wq_G/G/c ≈ Wq_M/M/c * (cv_a^2 + cv_s^2) / 2
    """
    if c <= 0 or service_rate <= 0 or arrival_rate <= 0:
        return 0.0
    rho = arrival_rate / (c * service_rate)
    if rho >= 1.0:
        return float('inf')

    # M/M/c Erlang-C 等待时间
    # P0 计算
    s = 0.0
    for k in range(c):
        s += (arrival_rate / service_rate) ** k / np.math.factorial(k)
    rho_c = arrival_rate / (c * service_rate)
    term = (arrival_rate / service_rate) ** c / np.math.factorial(c)
    p0 = 1.0 / (s + term / (1.0 - rho_c))
    pq = term * p0 / (1.0 - rho_c)
    wq_mmc = pq / (c * service_rate * (1.0 - rho))

    correction = (cv_arrival ** 2 + cv_service ** 2) / 2.0
    return wq_mmc * correction


# ---------------------------------------------------------------------------
# 方案对比
# ---------------------------------------------------------------------------
def build_comparison_table(results: List[SimulationResult]) -> pd.DataFrame:
    """将多个仿真结果汇总为方案对比表"""
    rows = []
    for r in results:
        cfg = r.config
        name = cfg.mode_name or cfg.line_config.mode.value
        total_machines = sum(s.machine_count for s in r.station_stats)
        avg_util = np.mean([s.utilization for s in r.station_stats])
        max_wait = max((s.avg_wait_time for s in r.station_stats), default=0.0)
        observation_weeks = max(1.0, cfg.sim_weeks - cfg.warmup_weeks)
        throughput_weekly = r.throughput.total_output / observation_weeks
        achievement = (throughput_weekly / cfg.line_config.target_weekly_output
                       if cfg.line_config.target_weekly_output else 0.0)
        oee = avg_util * 0.95 * 0.98  # 简化的 OEE = 利用率 × 性能 × 质量
        wip = _estimate_wip(r.station_stats)
        rows.append({
            "方案": name,
            "周吞吐量": round(throughput_weekly, 1),
            "目标达成率": f"{achievement:.1%}",
            "平均周期时间(min)": round(r.throughput.avg_cycle_time, 1),
            "平均在制品 WIP": round(wip, 1),
            "平均设备利用率": f"{avg_util:.1%}",
            "最大等待时间(min)": round(max_wait, 1),
            "OEE": f"{oee:.1%}",
            "设备总数": total_machines,
            "工人数": cfg.total_workers,
            "瓶颈工序": r.bottleneck.theoretical_bottleneck_name,
        })
    return pd.DataFrame(rows)


def _estimate_wip(station_stats: List[StationStats]) -> float:
    """估算在制品数量：队列 + 正在加工"""
    queue = sum(s.avg_queue_length for s in station_stats)
    running = sum(s.utilization * s.machine_count for s in station_stats)
    return queue + running


def scenario_radar_data(results: List[SimulationResult]) -> Tuple[List[str], Dict[str, List[float]]]:
    """生成雷达图所需的数据：维度列表 + 各方案归一化指标"""
    metrics = ["吞吐量", "利用率", "达成率", "OEE", "稳定性"]
    values: Dict[str, List[float]] = {}
    throughput_vals = []
    util_vals = []
    achieve_vals = []
    oee_vals = []
    stability_vals = []

    for r in results:
        cfg = r.config
        name = cfg.mode_name or cfg.line_config.mode.value
        observation_weeks = max(1.0, cfg.sim_weeks - cfg.warmup_weeks)
        tp = r.throughput.total_output / observation_weeks
        avg_util = np.mean([s.utilization for s in r.station_stats])
        achieve = (tp / cfg.line_config.target_weekly_output
                   if cfg.line_config.target_weekly_output else 0.0)
        oee = avg_util * 0.95 * 0.98
        # 稳定性：用各工序利用率标准差倒数归一化
        utils = [s.utilization for s in r.station_stats]
        stability = 1.0 / (1.0 + np.std(utils)) if utils else 0.0
        throughput_vals.append(tp)
        util_vals.append(avg_util)
        achieve_vals.append(achieve)
        oee_vals.append(oee)
        stability_vals.append(stability)

    def _norm(vals: List[float]) -> List[float]:
        m = max(vals) if max(vals) > 0 else 1.0
        return [v / m for v in vals]

    tp_n = _norm(throughput_vals)
    util_n = _norm(util_vals)
    achieve_n = _norm(achieve_vals)
    oee_n = _norm(oee_vals)
    stability_n = _norm(stability_vals)

    for i, r in enumerate(results):
        name = r.config.mode_name or r.config.line_config.mode.value
        values[name] = [tp_n[i], util_n[i], achieve_n[i], oee_n[i], stability_n[i]]

    return metrics, values


def compare_two_scenarios(baseline: SimulationResult,
                          alternative: SimulationResult) -> pd.DataFrame:
    """对比两个方案的关键指标差异"""
    def _extract(r: SimulationResult) -> Dict[str, float]:
        cfg = r.config
        observation_weeks = max(1.0, cfg.sim_weeks - cfg.warmup_weeks)
        tp = r.throughput.total_output / observation_weeks
        avg_util = np.mean([s.utilization for s in r.station_stats])
        achieve = (tp / cfg.line_config.target_weekly_output
                   if cfg.line_config.target_weekly_output else 0.0)
        oee = avg_util * 0.95 * 0.98
        wip = _estimate_wip(r.station_stats)
        return {
            "周吞吐量": tp,
            "平均利用率": avg_util,
            "目标达成率": achieve,
            "OEE": oee,
            "WIP": wip,
            "平均周期时间": r.throughput.avg_cycle_time,
        }

    b = _extract(baseline)
    a = _extract(alternative)
    rows = []
    for k in b:
        rows.append({
            "指标": k,
            baseline.config.mode_name or baseline.config.line_config.mode.value: round(b[k], 2),
            alternative.config.mode_name or alternative.config.line_config.mode.value: round(a[k], 2),
            "变化": f"{((a[k] - b[k]) / abs(b[k]) * 100) if b[k] != 0 else 0:.1f}%",
        })
    return pd.DataFrame(rows)
