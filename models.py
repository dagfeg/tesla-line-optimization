"""
models.py
基于瓶颈理论与离散事件仿真的汽车生产线优化平台 —— 数据模型与默认配置

本模块定义：
- 工序、设备、产线、仿真的数据类
- 默认产线参数（GA3 单线 / GA3+GA4 双线）
- 分布采样、工时换算等工具函数
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------
class ProcessType(Enum):
    """五大工艺段"""
    STAMPING = "冲压"
    WELDING = "焊装"
    PAINTING = "喷漆"
    ASSEMBLY = "总装"
    TESTING = "检测"


class DistributionType(Enum):
    """支持的加工/故障/维修时间分布"""
    CONSTANT = "constant"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRIANGULAR = "triangular"
    EXPONENTIAL = "exponential"


class LineMode(Enum):
    """产线组织模式"""
    GA3 = "GA3 单线模式"
    GA3_GA4 = "GA3 + GA4 双线模式"


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------
@dataclass
class ProcessConfig:
    """工序配置"""
    process_id: str
    name: str
    process_type: ProcessType
    sequence_index: int
    machine_count: int = 1
    processing_time_dist: DistributionType = DistributionType.LOGNORMAL
    processing_time_params: List[float] = field(default_factory=lambda: [5.0, 0.3])
    buffer_capacity: int = 5
    availability: float = 0.92
    setup_prob: float = 0.02
    setup_time_mean: float = 15.0
    quality_rate: float = 0.98
    rework_time: float = 10.0


@dataclass
class LineConfig:
    """生产线配置（含拓扑关系）"""
    line_id: str
    line_name: str
    mode: LineMode
    processes: List[ProcessConfig]
    # routing: 当前工序 -> 下游工序 id 列表；分支时取队列最短者
    routing: Dict[str, List[str]] = field(default_factory=dict)
    target_weekly_output: int = 5000


@dataclass
class SimulationConfig:
    """仿真运行配置"""
    line_config: LineConfig
    total_workers: int = 20
    work_hours_per_week: float = 100.0
    sim_weeks: float = 4.0
    warmup_weeks: float = 0.5
    arrival_interval_minutes: Optional[float] = None  # None 时按目标节拍计算
    seed: Optional[int] = 42
    mode_name: str = ""

    @property
    def minutes_per_week(self) -> float:
        return self.work_hours_per_week * 60.0

    @property
    def sim_time(self) -> float:
        return self.sim_weeks * self.minutes_per_week

    @property
    def warmup_time(self) -> float:
        return self.warmup_weeks * self.minutes_per_week

    def get_arrival_interval(self) -> float:
        if self.arrival_interval_minutes is not None:
            return max(0.1, self.arrival_interval_minutes)
        # 按目标产出计算理论节拍（分钟/辆）
        return max(0.1, self.minutes_per_week / self.line_config.target_weekly_output)


@dataclass
class StationStats:
    """单个工序的统计结果"""
    process_id: str
    name: str
    process_type: ProcessType
    machine_count: int
    worker_count: int
    buffer_capacity: int
    theoretical_capacity: float  # 件/周
    utilization: float           # 运行时间占比
    idle_ratio: float
    blocked_ratio: float
    breakdown_ratio: float
    avg_queue_length: float
    max_queue_length: int
    avg_wait_time: float         # 分钟
    avg_service_time: float      # 分钟
    blocked_probability: float   # 曾遭遇阻塞的工件比例
    total_processed: int
    total_blocked: int
    queue_time_series: List[Tuple[float, int]] = field(default_factory=list)
    wait_times: List[float] = field(default_factory=list)


@dataclass
class ThroughputStats:
    """产出统计"""
    total_output: int
    weekly_output: List[int]
    cumulative_output: List[int]
    avg_cycle_time: float
    cycle_time_var: float
    wip_mean: float


@dataclass
class BottleneckReport:
    """瓶颈分析报告"""
    theoretical_bottleneck_id: str
    theoretical_bottleneck_name: str
    capacity_bottleneck_id: str
    utilization_bottleneck_id: str
    utilization_bottleneck_name: str
    queue_bottleneck_id: str
    queue_bottleneck_name: str
    ranked_stations: List[Dict]
    drum_rate: float
    time_buffer: float


@dataclass
class SimulationResult:
    """单次仿真结果"""
    config: SimulationConfig
    station_stats: List[StationStats]
    throughput: ThroughputStats
    bottleneck: BottleneckReport
    sim_duration_min: float
    observation_min: float


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def sample_time(rng: np.random.Generator,
                dist: DistributionType,
                params: List[float]) -> float:
    """从给定分布中采样一个正的处理时间（分钟）"""
    if dist == DistributionType.CONSTANT:
        return max(0.05, params[0])
    if dist == DistributionType.EXPONENTIAL:
        return max(0.05, rng.exponential(params[0]))
    if dist == DistributionType.NORMAL:
        mu, sigma = params[0], params[1]
        return max(0.05, rng.normal(mu, sigma))
    if dist == DistributionType.LOGNORMAL:
        # params: [mean, sigma] 指对数正态的参数 mu/sigma
        mu, sigma = params[0], params[1]
        return max(0.05, rng.lognormal(mu, sigma))
    if dist == DistributionType.TRIANGULAR:
        low, mode, high = params[0], params[1], params[2]
        return max(0.05, rng.triangular(low, mode, high))
    return max(0.05, params[0])


def mean_time(dist: DistributionType, params: List[float]) -> float:
    """计算分布的理论均值（用于产能估算）"""
    if dist == DistributionType.CONSTANT:
        return params[0]
    if dist == DistributionType.EXPONENTIAL:
        return params[0]
    if dist == DistributionType.NORMAL:
        return params[0]
    if dist == DistributionType.LOGNORMAL:
        mu, sigma = params[0], params[1]
        return np.exp(mu + 0.5 * sigma ** 2)
    if dist == DistributionType.TRIANGULAR:
        low, mode, high = params[0], params[1], params[2]
        return (low + mode + high) / 3.0
    return params[0]


def allocate_workers(processes: List[ProcessConfig], total_workers: int) -> Dict[str, int]:
    """按并行设备数比例把总工人数分配到各工序，至少 1 人"""
    total_machines = sum(p.machine_count for p in processes)
    allocation: Dict[str, int] = {}
    if total_workers <= 0:
        for p in processes:
            allocation[p.process_id] = 1
        return allocation
    for p in processes:
        share = max(1, int(round(p.machine_count * total_workers / total_machines)))
        allocation[p.process_id] = share
    # 如果四舍五入后超过总数，从人数最多的工序扣减
    while sum(allocation.values()) > total_workers and total_workers > 0:
        max_id = max(allocation, key=lambda k: allocation[k])
        if allocation[max_id] > 1:
            allocation[max_id] -= 1
    return allocation


# ---------------------------------------------------------------------------
# 默认产线配置
# ---------------------------------------------------------------------------
def _base_processes() -> List[ProcessConfig]:
    """GA3 单线的五道工序默认参数"""
    return [
        ProcessConfig(
            process_id="stamping",
            name="冲压",
            process_type=ProcessType.STAMPING,
            sequence_index=0,
            machine_count=2,
            processing_time_dist=DistributionType.LOGNORMAL,
            processing_time_params=[1.35, 0.25],   # 均值约 4 min
            buffer_capacity=6,
            availability=0.94,
            setup_prob=0.02,
            setup_time_mean=12.0,
            quality_rate=0.985,
            rework_time=8.0,
        ),
        ProcessConfig(
            process_id="welding",
            name="焊装",
            process_type=ProcessType.WELDING,
            sequence_index=1,
            machine_count=3,
            processing_time_dist=DistributionType.LOGNORMAL,
            processing_time_params=[1.75, 0.30],   # 均值约 6 min
            buffer_capacity=8,
            availability=0.90,
            setup_prob=0.03,
            setup_time_mean=18.0,
            quality_rate=0.97,
            rework_time=12.0,
        ),
        ProcessConfig(
            process_id="painting",
            name="喷漆",
            process_type=ProcessType.PAINTING,
            sequence_index=2,
            machine_count=2,
            processing_time_dist=DistributionType.LOGNORMAL,
            processing_time_params=[1.55, 0.20],   # 均值约 4.8 min
            buffer_capacity=6,
            availability=0.92,
            setup_prob=0.05,
            setup_time_mean=25.0,
            quality_rate=0.96,
            rework_time=15.0,
        ),
        ProcessConfig(
            process_id="assembly",
            name="总装",
            process_type=ProcessType.ASSEMBLY,
            sequence_index=3,
            machine_count=4,
            processing_time_dist=DistributionType.LOGNORMAL,
            processing_time_params=[2.05, 0.22],   # 均值约 8 min
            buffer_capacity=10,
            availability=0.93,
            setup_prob=0.01,
            setup_time_mean=10.0,
            quality_rate=0.98,
            rework_time=10.0,
        ),
        ProcessConfig(
            process_id="testing",
            name="检测",
            process_type=ProcessType.TESTING,
            sequence_index=4,
            machine_count=2,
            processing_time_dist=DistributionType.LOGNORMAL,
            processing_time_params=[1.05, 0.18],   # 均值约 3 min
            buffer_capacity=5,
            availability=0.95,
            setup_prob=0.005,
            setup_time_mean=5.0,
            quality_rate=0.99,
            rework_time=5.0,
        ),
    ]


def build_ga3_config(machine_counts: Optional[Dict[str, int]] = None,
                     workers: int = 20,
                     work_hours: float = 100.0) -> LineConfig:
    """构建 GA3 单线配置"""
    processes = _base_processes()
    if machine_counts:
        for p in processes:
            if p.process_id in machine_counts:
                p.machine_count = max(1, machine_counts[p.process_id])
    routing = {}
    for i, p in enumerate(processes[:-1]):
        routing[p.process_id] = [processes[i + 1].process_id]
    return LineConfig(
        line_id="line_ga3",
        line_name="GA3 总装线",
        mode=LineMode.GA3,
        processes=processes,
        routing=routing,
        target_weekly_output=5000,
    )


def build_ga3_ga4_config(machine_counts: Optional[Dict[str, int]] = None,
                         workers: int = 35,
                         work_hours: float = 100.0,
                         assembly_lines: int = 2) -> LineConfig:
    """
    构建 GA3 + GA4 配置：
    - 上游共享冲压、焊装、喷漆
    - 下游并行 assembly_lines 条总装+检测线
    """
    upstream = [
        ProcessConfig(
            process_id="stamping",
            name="冲压",
            process_type=ProcessType.STAMPING,
            sequence_index=0,
            machine_count=3,
            processing_time_dist=DistributionType.LOGNORMAL,
            processing_time_params=[1.35, 0.25],
            buffer_capacity=8,
            availability=0.94,
            setup_prob=0.02,
            setup_time_mean=12.0,
            quality_rate=0.985,
            rework_time=8.0,
        ),
        ProcessConfig(
            process_id="welding",
            name="焊装",
            process_type=ProcessType.WELDING,
            sequence_index=1,
            machine_count=5,
            processing_time_dist=DistributionType.LOGNORMAL,
            processing_time_params=[1.75, 0.30],
            buffer_capacity=12,
            availability=0.90,
            setup_prob=0.03,
            setup_time_mean=18.0,
            quality_rate=0.97,
            rework_time=12.0,
        ),
        ProcessConfig(
            process_id="painting",
            name="喷漆",
            process_type=ProcessType.PAINTING,
            sequence_index=2,
            machine_count=4,
            processing_time_dist=DistributionType.LOGNORMAL,
            processing_time_params=[1.55, 0.20],
            buffer_capacity=10,
            availability=0.92,
            setup_prob=0.05,
            setup_time_mean=25.0,
            quality_rate=0.96,
            rework_time=15.0,
        ),
    ]
    processes: List[ProcessConfig] = []
    processes.extend(upstream)
    routing: Dict[str, List[str]] = {}
    for i, p in enumerate(upstream[:-1]):
        routing[p.process_id] = [upstream[i + 1].process_id]

    # 下游并行总装/检测线
    paint_id = upstream[-1].process_id
    downstream_ids: List[str] = []
    for line_idx in range(assembly_lines):
        suffix = f"_L{line_idx + 1}"
        asm = ProcessConfig(
            process_id=f"assembly{suffix}",
            name=f"总装{suffix}",
            process_type=ProcessType.ASSEMBLY,
            sequence_index=3,
            machine_count=4,
            processing_time_dist=DistributionType.LOGNORMAL,
            processing_time_params=[2.05, 0.22],
            buffer_capacity=8,
            availability=0.93,
            setup_prob=0.01,
            setup_time_mean=10.0,
            quality_rate=0.98,
            rework_time=10.0,
        )
        test = ProcessConfig(
            process_id=f"testing{suffix}",
            name=f"检测{suffix}",
            process_type=ProcessType.TESTING,
            sequence_index=4,
            machine_count=2,
            processing_time_dist=DistributionType.LOGNORMAL,
            processing_time_params=[1.05, 0.18],
            buffer_capacity=4,
            availability=0.95,
            setup_prob=0.005,
            setup_time_mean=5.0,
            quality_rate=0.99,
            rework_time=5.0,
        )
        processes.extend([asm, test])
        routing[paint_id] = routing.get(paint_id, []) + [asm.process_id]
        routing[asm.process_id] = [test.process_id]
        downstream_ids.append(test.process_id)

    if machine_counts:
        for p in processes:
            if p.process_id in machine_counts:
                p.machine_count = max(1, machine_counts[p.process_id])

    return LineConfig(
        line_id="line_ga3_ga4",
        line_name=f"GA3 + GA4 双线模式（{assembly_lines} 条总装线）",
        mode=LineMode.GA3_GA4,
        processes=processes,
        routing=routing,
        target_weekly_output=5000 * assembly_lines,
    )
