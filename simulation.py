"""
simulation.py
基于 SimPy 的离散事件仿真引擎

核心设计：
- 每道工序配置一个输入缓冲区（SimPy Store）
- 每道工序的每台设备对应一个独立的服务进程（machine server）
- 工序内工人作为共享资源，服务前需同时占用 1 名工人和 1 台设备
- 服务完成后释放工人；若下游缓冲区满，设备进入 BLOCKED 状态
- 通过状态面积法实时统计利用率、队列长度、等待时间等指标
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Callable, Optional
import itertools
import numpy as np
import simpy

from models import (
    ProcessConfig,
    LineConfig,
    SimulationConfig,
    SimulationResult,
    StationStats,
    ThroughputStats,
    BottleneckReport,
    ProcessType,
    DistributionType,
    sample_time,
    mean_time,
    allocate_workers,
)
import analysis


# ---------------------------------------------------------------------------
# 内部运行时对象
# ---------------------------------------------------------------------------
class _StationRuntime:
    """仿真运行时单个工序的 SimPy 对象集合"""

    def __init__(self, env: simpy.Environment, config: ProcessConfig,
                 worker_capacity: int):
        self.config = config
        self.buffer = simpy.Store(env, capacity=config.buffer_capacity)
        self.worker = simpy.Resource(env, capacity=worker_capacity)


class _StationMonitor:
    """统计采集器：状态面积法 + 队列面积法"""

    def __init__(self, env: simpy.Environment, config: ProcessConfig,
                 worker_count: int, warmup_time: float):
        self.env = env
        self.config = config
        self.worker_count = worker_count
        self.warmup_time = warmup_time
        self.machine_count = config.machine_count

        self.last_t = env.now
        # OUT 表示尚未启动的服务进程，面积最终应为 0
        self.counts = {"OUT": config.machine_count, "IDLE": 0,
                       "RUNNING": 0, "BLOCKED": 0, "BREAKDOWN": 0}
        self.areas = {k: 0.0 for k in self.counts}

        self.queue_len = 0
        self.queue_area = 0.0
        self.max_queue = 0
        self.queue_ts: List[Tuple[float, int]] = []

        self.wait_times: List[float] = []
        self.service_times: List[float] = []
        self.blocked_count = 0
        self.processed_count = 0

    def _update(self):
        """从上一次记录点推进到当前时刻，累加面积"""
        now = self.env.now
        dt = now - self.last_t
        if dt > 0:
            for state in self.counts:
                self.areas[state] += self.counts[state] * dt
            self.queue_area += self.queue_len * dt
            if now >= self.warmup_time:
                self.queue_ts.append((now, self.queue_len))
            self.last_t = now

    def transition(self, old_state: str, new_state: str):
        """状态迁移：减少旧状态计数，增加新状态计数"""
        self._update()
        self.counts[old_state] -= 1
        self.counts[new_state] += 1

    def set_queue(self, length: int):
        """更新队列长度"""
        self._update()
        self.queue_len = length
        self.max_queue = max(self.max_queue, length)

    def record_wait(self, wait: float):
        if self.env.now >= self.warmup_time:
            self.wait_times.append(wait)

    def record_service(self, service: float):
        if self.env.now >= self.warmup_time:
            self.service_times.append(service)

    def record_processed(self, was_blocked: bool):
        if self.env.now >= self.warmup_time:
            self.processed_count += 1
            if was_blocked:
                self.blocked_count += 1

    def reset(self):
        """预热期结束后重置累计量，只保留当前状态计数"""
        self._update()
        self.last_t = self.env.now
        for k in self.areas:
            self.areas[k] = 0.0
        self.queue_area = 0.0
        self.max_queue = self.queue_len
        self.queue_ts = []
        self.wait_times = []
        self.service_times = []
        self.blocked_count = 0
        self.processed_count = 0


# ---------------------------------------------------------------------------
# SimPy 进程
# ---------------------------------------------------------------------------
def _machine_server(env: simpy.Environment,
                    station: _StationRuntime,
                    all_stations: Dict[str, _StationRuntime],
                    monitors: Dict[str, _StationMonitor],
                    downstream_ids: List[str],
                    monitor: _StationMonitor,
                    rng: np.random.Generator,
                    completion_log: Callable[[Dict, float], None]):
    """单台设备的服务进程"""
    state = "OUT"
    monitor.transition("OUT", "IDLE")
    state = "IDLE"

    while True:
        # 1. 从缓冲区取件
        part = yield station.buffer.get()
        monitor.set_queue(len(station.buffer.items))

        # 2. 申请工人
        worker_req = station.worker.request()
        yield worker_req

        wait = env.now - part.get("enter_time", env.now)
        monitor.record_wait(wait)

        # 3. 开始加工
        monitor.transition(state, "RUNNING")
        state = "RUNNING"

        cfg = station.config
        # 基础加工时间
        proc = sample_time(rng, cfg.processing_time_dist, cfg.processing_time_params)

        # 换模时间
        if rng.random() < cfg.setup_prob:
            proc += max(0.5, rng.exponential(cfg.setup_time_mean))

        # 质量返工
        if rng.random() > cfg.quality_rate:
            proc += cfg.rework_time

        # 设备故障（非抢占式：将停机叠加到本次服务中）
        had_breakdown = False
        if rng.random() > cfg.availability:
            had_breakdown = True
            monitor.transition("RUNNING", "BREAKDOWN")
            state = "BREAKDOWN"
            # 维修时间对数正态，均值约 15 min
            repair = sample_time(rng, DistributionType.LOGNORMAL, [2.5, 0.5])
            yield env.timeout(repair)
            monitor.transition("BREAKDOWN", "RUNNING")
            state = "RUNNING"

        yield env.timeout(proc)
        monitor.record_service(proc)

        # 4. 释放工人（工人不随车移动）
        station.worker.release(worker_req)

        # 5. 转往下一道工序；若缓冲区满则设备进入 BLOCKED
        was_blocked = False
        if downstream_ids:
            next_id = _choose_downstream(all_stations, downstream_ids)
            next_buffer = all_stations[next_id].buffer
            monitor.transition(state, "BLOCKED")
            state = "BLOCKED"
            was_blocked = True
            yield next_buffer.put(part)
            # 进入下游缓冲区时更新进入时间
            part["enter_time"] = env.now
            monitor.set_queue(len(station.buffer.items))
            monitors[next_id].set_queue(len(next_buffer.items))
            monitor.transition("BLOCKED", "IDLE")
            state = "IDLE"
        else:
            # 最终检测完成
            completion_log(part, env.now)
            monitor.transition(state, "IDLE")
            state = "IDLE"

        monitor.record_processed(was_blocked)


def _choose_downstream(all_stations: Dict[str, _StationRuntime],
                       downstream_ids: List[str]) -> str:
    """分支工位选择队列最短的下游"""
    if len(downstream_ids) == 1:
        return downstream_ids[0]
    best = min(downstream_ids,
               key=lambda sid: len(all_stations[sid].buffer.items))
    return best


def _arrival_generator(env: simpy.Environment,
                       first_buffer: simpy.Store,
                       first_monitor: _StationMonitor,
                       interval: float,
                       rng: np.random.Generator,
                       sim_time: float,
                       id_gen: itertools.count):
    """毛坯到达发生器，按泊松过程投料"""
    while env.now < sim_time:
        yield env.timeout(max(0.05, rng.exponential(interval)))
        if env.now >= sim_time:
            break
        part = {
            "id": next(id_gen),
            "start_time": env.now,
            "enter_time": env.now,
        }
        yield first_buffer.put(part)
        first_monitor.set_queue(len(first_buffer.items))


def _reset_monitors_after_warmup(env: simpy.Environment,
                                 monitors: Dict[str, _StationMonitor],
                                 warmup_time: float):
    """预热结束后重置统计器"""
    yield env.timeout(warmup_time)
    for monitor in monitors.values():
        monitor.reset()


# ---------------------------------------------------------------------------
# 仿真入口与结果构造
# ---------------------------------------------------------------------------
def run_simulation(config: SimulationConfig) -> SimulationResult:
    """运行一次离散事件仿真并返回完整结果"""
    env = simpy.Environment()
    rng = np.random.default_rng(config.seed)
    line = config.line_config

    # 分配工人
    worker_alloc = allocate_workers(line.processes, config.total_workers)

    # 创建运行时工序对象和监控器
    stations: Dict[str, _StationRuntime] = {}
    monitors: Dict[str, _StationMonitor] = {}
    for proc_cfg in line.processes:
        st = _StationRuntime(env, proc_cfg, worker_alloc[proc_cfg.process_id])
        stations[proc_cfg.process_id] = st
        monitors[proc_cfg.process_id] = _StationMonitor(
            env, proc_cfg, worker_alloc[proc_cfg.process_id], config.warmup_time)

    # 启动设备服务进程
    for proc_cfg in line.processes:
        downstream_ids = line.routing.get(proc_cfg.process_id, [])
        st = stations[proc_cfg.process_id]
        mon = monitors[proc_cfg.process_id]
        for _ in range(proc_cfg.machine_count):
            env.process(_machine_server(
                env, st, stations, monitors, downstream_ids, mon, rng,
                lambda part, t: _record_completion(part, t)))

    # 到达发生器
    all_downstream = set()
    for ids in line.routing.values():
        all_downstream.update(ids)
    first_ids = [p.process_id for p in line.processes if p.process_id not in all_downstream]
    first_id = first_ids[0] if first_ids else line.processes[0].process_id
    first_buffer = stations[first_id].buffer

    completions: List[Tuple[float, float]] = []
    cycle_times: List[float] = []

    def _record_completion(part: Dict, t: float):
        completions.append((t, t - part.get("start_time", t)))
        if t >= config.warmup_time:
            cycle_times.append(t - part.get("start_time", t))

    env.process(_arrival_generator(
        env, first_buffer, monitors[first_id], config.get_arrival_interval(),
        rng, config.sim_time, itertools.count(1)))

    # 预热结束后重置统计器
    env.process(_reset_monitors_after_warmup(env, monitors, config.warmup_time))

    # 运行仿真
    env.run(until=config.sim_time)

    # 构造统计对象
    observation_time = max(1.0, config.sim_time - config.warmup_time)
    station_stats = _build_station_stats(
        line.processes, stations, monitors, worker_alloc, config, observation_time)
    throughput = _build_throughput_stats(completions, cycle_times, config)
    bottleneck = analysis.analyze_bottleneck(station_stats, throughput, config)

    return SimulationResult(
        config=config,
        station_stats=station_stats,
        throughput=throughput,
        bottleneck=bottleneck,
        sim_duration_min=config.sim_time,
        observation_min=observation_time,
    )


def _build_station_stats(processes: List[ProcessConfig],
                         stations: Dict[str, _StationRuntime],
                         monitors: Dict[str, _StationMonitor],
                         worker_alloc: Dict[str, int],
                         config: SimulationConfig,
                         observation_time: float) -> List[StationStats]:
    """由监控器构造工序统计列表"""
    stats: List[StationStats] = []
    for cfg in processes:
        mon = monitors[cfg.process_id]
        mon._update()  # 结算到仿真结束
        machine_seconds = cfg.machine_count * observation_time

        utilization = mon.areas["RUNNING"] / machine_seconds if machine_seconds else 0.0
        idle_ratio = mon.areas["IDLE"] / machine_seconds if machine_seconds else 0.0
        blocked_ratio = mon.areas["BLOCKED"] / machine_seconds if machine_seconds else 0.0
        breakdown_ratio = mon.areas["BREAKDOWN"] / machine_seconds if machine_seconds else 0.0

        avg_queue = mon.queue_area / observation_time if observation_time else 0.0
        avg_wait = float(np.mean(mon.wait_times)) if mon.wait_times else 0.0
        avg_service = float(np.mean(mon.service_times)) if mon.service_times else 0.0
        blocked_prob = (mon.blocked_count / mon.processed_count
                        if mon.processed_count else 0.0)

        # 理论产能（件/周）
        mean_proc = mean_time(cfg.processing_time_dist, cfg.processing_time_params)
        theoretical_capacity = (config.work_hours_per_week * 60.0 *
                                cfg.availability * cfg.machine_count / mean_proc)

        stats.append(StationStats(
            process_id=cfg.process_id,
            name=cfg.name,
            process_type=cfg.process_type,
            machine_count=cfg.machine_count,
            worker_count=worker_alloc[cfg.process_id],
            buffer_capacity=cfg.buffer_capacity,
            theoretical_capacity=theoretical_capacity,
            utilization=utilization,
            idle_ratio=idle_ratio,
            blocked_ratio=blocked_ratio,
            breakdown_ratio=breakdown_ratio,
            avg_queue_length=avg_queue,
            max_queue_length=mon.max_queue,
            avg_wait_time=avg_wait,
            avg_service_time=avg_service,
            blocked_probability=blocked_prob,
            total_processed=mon.processed_count,
            total_blocked=mon.blocked_count,
            queue_time_series=mon.queue_ts,
            wait_times=mon.wait_times,
        ))
    return stats


def _build_throughput_stats(completions: List[Tuple[float, float]],
                            cycle_times: List[float],
                            config: SimulationConfig) -> ThroughputStats:
    """构造产出统计"""
    week_min = config.minutes_per_week
    n_weeks = max(1, int(np.ceil(config.sim_time / week_min)))
    weekly = [0] * n_weeks
    for t, _ in completions:
        w = int(t // week_min)
        if 0 <= w < n_weeks:
            weekly[w] += 1

    cumulative = list(np.cumsum(weekly))
    total = sum(weekly)

    avg_cycle = float(np.mean(cycle_times)) if cycle_times else 0.0
    cycle_var = float(np.var(cycle_times)) if cycle_times else 0.0

    # WIP 均值在分析阶段通过各工序平均在制数汇总
    wip_mean = 0.0

    return ThroughputStats(
        total_output=total,
        weekly_output=weekly,
        cumulative_output=cumulative,
        avg_cycle_time=avg_cycle,
        cycle_time_var=cycle_var,
        wip_mean=wip_mean,
    )
