from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


EdgeKey = Tuple[str, str, str]
Path = List[EdgeKey]


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    name: str
    alpha: float
    beta: float
    power: int = 4

    def latency(self, load: float) -> float:
        return self.alpha * (load ** self.power) + self.beta

    def total_cost(self, load: float) -> float:
        return load * self.latency(load)


class CongestionGraph:
    def __init__(self) -> None:
        self._edges: Dict[EdgeKey, Edge] = {}
        self._adj: Dict[str, List[EdgeKey]] = {}

    def add_edge(
        self,
        src: str,
        dst: str,
        *,
        alpha: float,
        beta: float,
        name: Optional[str] = None,
        power: int = 4,
    ) -> EdgeKey:
        edge_name = name or f"{src}->{dst}"
        key = (src, dst, edge_name)
        edge = Edge(src=src, dst=dst, name=edge_name, alpha=alpha, beta=beta, power=power)
        self._edges[key] = edge
        self._adj.setdefault(src, []).append(key)
        self._adj.setdefault(dst, [])
        return key

    def edges(self) -> Dict[EdgeKey, Edge]:
        return self._edges

    def neighbors(self, node: str) -> Sequence[EdgeKey]:
        return self._adj.get(node, [])

    def copy_without_edge(self, edge_key: EdgeKey) -> "CongestionGraph":
        clone = CongestionGraph()
        for key, edge in self._edges.items():
            if key == edge_key:
                continue
            clone.add_edge(
                edge.src,
                edge.dst,
                alpha=edge.alpha,
                beta=edge.beta,
                name=edge.name,
                power=edge.power,
            )
        return clone


def dijkstra_path(
    graph: CongestionGraph,
    source: str,
    sink: str,
    loads: Dict[EdgeKey, int],
    *,
    own_increment: int = 1,
    perceived_loads: Optional[Dict[EdgeKey, float]] = None,
) -> Path:
    dist: Dict[str, float] = {source: 0.0}
    parent: Dict[str, Tuple[str, EdgeKey]] = {}
    pq: List[Tuple[float, str]] = [(0.0, source)]

    while pq:
        cur_d, u = heapq.heappop(pq)
        if cur_d > dist.get(u, math.inf):
            continue
        if u == sink:
            break
        for edge_key in graph.neighbors(u):
            edge = graph.edges()[edge_key]
            base_load = (
                perceived_loads.get(edge_key, 0.0)
                if perceived_loads is not None
                else float(loads.get(edge_key, 0))
            )
            w = edge.latency(max(0.0, base_load + own_increment))
            nd = cur_d + w
            if nd < dist.get(edge.dst, math.inf):
                dist[edge.dst] = nd
                parent[edge.dst] = (u, edge_key)
                heapq.heappush(pq, (nd, edge.dst))

    if sink not in parent and sink != source:
        return []

    path: Path = []
    node = sink
    while node != source:
        prev, edge_key = parent[node]
        path.append(edge_key)
        node = prev
    path.reverse()
    return path


def path_latency(graph: CongestionGraph, path: Path, loads: Dict[EdgeKey, int]) -> float:
    return sum(graph.edges()[e].latency(loads[e]) for e in path)


def social_cost(graph: CongestionGraph, loads: Dict[EdgeKey, int]) -> float:
    return sum(edge.total_cost(loads.get(k, 0)) for k, edge in graph.edges().items())


def _enumerate_simple_paths(
    graph: CongestionGraph,
    source: str,
    sink: str,
    *,
    max_depth: int = 8,
    max_paths: int = 64,
) -> List[Path]:
    out: List[Path] = []

    def dfs(node: str, visited: set[str], acc: Path) -> None:
        if len(out) >= max_paths or len(acc) > max_depth:
            return
        if node == sink:
            out.append(list(acc))
            return
        for edge_key in graph.neighbors(node):
            nxt = graph.edges()[edge_key].dst
            if nxt in visited:
                continue
            visited.add(nxt)
            acc.append(edge_key)
            dfs(nxt, visited, acc)
            acc.pop()
            visited.remove(nxt)

    dfs(source, {source}, [])
    return out


def iterative_best_response(
    graph: CongestionGraph,
    source: str,
    sink: str,
    n_agents: int,
    *,
    max_iters: int = 200,
    limited_information: bool = False,
    information_noise: float = 0.0,
    adversarial_fraction: float = 0.0,
    seed: int = 0,
) -> Tuple[List[Path], Dict[EdgeKey, int], bool]:
    rng = random.Random(seed)
    agents: List[Path] = [dijkstra_path(graph, source, sink, {}) for _ in range(n_agents)]
    loads: Dict[EdgeKey, int] = {k: 0 for k in graph.edges().keys()}
    for p in agents:
        for e in p:
            loads[e] += 1

    adversarial_count = int(round(n_agents * adversarial_fraction))
    adversarial = set(range(adversarial_count))
    memories: List[Dict[EdgeKey, float]] = [{k: 0.0 for k in graph.edges().keys()} for _ in range(n_agents)]

    for _ in range(max_iters):
        changed = False
        for i in range(n_agents):
            old = agents[i]
            for e in old:
                loads[e] -= 1

            perceived_loads = None
            if limited_information:
                memory = memories[i]
                perceived_loads = {}
                for k in graph.edges().keys():
                    observed = memory[k]
                    noisy = observed + rng.uniform(-information_noise, information_noise)
                    perceived_loads[k] = max(0.0, noisy)

            if i in adversarial:
                candidates = _enumerate_simple_paths(graph, source, sink)
                if not candidates:
                    new_path = []
                else:
                    new_path = max(
                        candidates,
                        key=lambda p: sum(
                            graph.edges()[e].latency(loads[e] + 1) for e in p
                        ),
                    )
            else:
                new_path = dijkstra_path(
                    graph,
                    source,
                    sink,
                    loads,
                    own_increment=1,
                    perceived_loads=perceived_loads,
                )

            agents[i] = new_path
            for e in new_path:
                loads[e] += 1

            if limited_information:
                mem = memories[i]
                for e in new_path:
                    mem[e] = float(loads[e])

            if new_path != old:
                changed = True

        if not changed:
            return agents, loads, True

    return agents, loads, False


def worst_nash_equilibrium_cost(
    graph: CongestionGraph,
    source: str,
    sink: str,
    n_agents: int,
    *,
    restarts: int = 5,
    **kwargs: object,
) -> float:
    worst = -math.inf
    for s in range(restarts):
        _, loads, _ = iterative_best_response(
            graph,
            source,
            sink,
            n_agents,
            seed=s,
            **kwargs,
        )
        worst = max(worst, social_cost(graph, loads))
    return worst


def min_cost_max_flow_social_optimum(
    graph: CongestionGraph,
    source: str,
    sink: str,
    n_agents: int,
) -> float:
    nodes = sorted({source, sink, *[e.src for e in graph.edges().values()], *[e.dst for e in graph.edges().values()]})
    node_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    class Arc:
        __slots__ = ("to", "rev", "cap", "cost")

        def __init__(self, to: int, rev: int, cap: int, cost: float) -> None:
            self.to = to
            self.rev = rev
            self.cap = cap
            self.cost = cost

    g: List[List[Arc]] = [[] for _ in range(N)]

    def add_arc(u: int, v: int, cap: int, cost: float) -> None:
        g[u].append(Arc(v, len(g[v]), cap, cost))
        g[v].append(Arc(u, len(g[u]) - 1, 0, -cost))

    for edge in graph.edges().values():
        u = node_idx[edge.src]
        v = node_idx[edge.dst]
        for i in range(1, n_agents + 1):
            c_i = edge.total_cost(float(i))
            c_prev = edge.total_cost(float(i - 1))
            marginal = c_i - c_prev
            add_arc(u, v, 1, marginal)

    s = node_idx[source]
    t = node_idx[sink]
    flow = 0
    total = 0.0
    potential = [0.0] * N

    while flow < n_agents:
        dist = [math.inf] * N
        parent_node = [-1] * N
        parent_edge = [-1] * N
        dist[s] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, s)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for ei, arc in enumerate(g[u]):
                if arc.cap <= 0:
                    continue
                nd = d + arc.cost + potential[u] - potential[arc.to]
                if nd < dist[arc.to]:
                    dist[arc.to] = nd
                    parent_node[arc.to] = u
                    parent_edge[arc.to] = ei
                    heapq.heappush(pq, (nd, arc.to))

        if dist[t] == math.inf:
            raise ValueError("No feasible flow for all agents")

        for i in range(N):
            if dist[i] < math.inf:
                potential[i] += dist[i]

        add = n_agents - flow
        v = t
        while v != s:
            u = parent_node[v]
            ei = parent_edge[v]
            add = min(add, g[u][ei].cap)
            v = u

        v = t
        while v != s:
            u = parent_node[v]
            ei = parent_edge[v]
            arc = g[u][ei]
            arc.cap -= add
            g[v][arc.rev].cap += add
            total += add * arc.cost
            v = u

        flow += add

    return total


def price_of_anarchy(
    graph: CongestionGraph,
    source: str,
    sink: str,
    n_agents: int,
    *,
    restarts: int = 5,
    **kwargs: object,
) -> float:
    worst_ne = worst_nash_equilibrium_cost(
        graph,
        source,
        sink,
        n_agents,
        restarts=restarts,
        **kwargs,
    )
    optimum = min_cost_max_flow_social_optimum(graph, source, sink, n_agents)
    if optimum == 0:
        return 1.0
    return worst_ne / optimum


def frank_wolfe_equilibrium(
    graph: CongestionGraph,
    source: str,
    sink: str,
    demand: float,
    *,
    max_iters: int = 100,
    tolerance: float = 1e-6,
) -> Dict[EdgeKey, float]:
    flow: Dict[EdgeKey, float] = {k: 0.0 for k in graph.edges().keys()}
    initial_path = dijkstra_path(graph, source, sink, {k: 0 for k in graph.edges().keys()}, own_increment=0)
    for e in initial_path:
        flow[e] += demand

    def objective(cur: Dict[EdgeKey, float]) -> float:
        val = 0.0
        for k, edge in graph.edges().items():
            x = max(0.0, cur[k])
            val += edge.alpha * (x ** (edge.power + 1)) / (edge.power + 1) + edge.beta * x
        return val

    for _ in range(max_iters):
        shortest = dijkstra_path(graph, source, sink, {k: int(round(v)) for k, v in flow.items()}, own_increment=0)
        y = {k: 0.0 for k in flow.keys()}
        for e in shortest:
            y[e] += demand

        direction = {k: y[k] - flow[k] for k in flow.keys()}
        gap = sum(abs(v) for v in direction.values())
        if gap < tolerance:
            break

        lo, hi = 0.0, 1.0
        for _ in range(40):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            f1 = objective({k: flow[k] + m1 * direction[k] for k in flow.keys()})
            f2 = objective({k: flow[k] + m2 * direction[k] for k in flow.keys()})
            if f1 <= f2:
                hi = m2
            else:
                lo = m1
        gamma = (lo + hi) / 2.0

        new_flow = {k: flow[k] + gamma * direction[k] for k in flow.keys()}
        delta = sum(abs(new_flow[k] - flow[k]) for k in flow.keys())
        flow = new_flow
        if delta < tolerance:
            break

    return flow


def barabasi_albert_graph(
    n_nodes: int,
    m: int,
    *,
    alpha: float,
    beta: float,
    power: int = 4,
    bidirectional: bool = True,
    seed: int = 0,
) -> CongestionGraph:
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    if m < 1 or m >= n_nodes:
        raise ValueError("m must satisfy 1 <= m < n_nodes")

    rng = random.Random(seed)
    graph = CongestionGraph()
    degrees: Dict[int, int] = {}
    existing = set()

    for i in range(m + 1):
        degrees[i] = 0
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            existing.add((i, j))
            degrees[i] += 1
            degrees[j] += 1
            graph.add_edge(str(i), str(j), alpha=alpha, beta=beta, power=power)
            if bidirectional:
                graph.add_edge(str(j), str(i), alpha=alpha, beta=beta, power=power)

    for new_node in range(m + 1, n_nodes):
        degrees[new_node] = 0
        total_degree = sum(degrees.values())
        targets = set()
        while len(targets) < m:
            r = rng.uniform(0, total_degree)
            upto = 0.0
            chosen = 0
            for node, deg in degrees.items():
                upto += deg
                if upto >= r:
                    chosen = node
                    break
            if chosen != new_node:
                targets.add(chosen)

        for t in targets:
            a, b = min(new_node, t), max(new_node, t)
            if (a, b) in existing:
                continue
            existing.add((a, b))
            degrees[new_node] += 1
            degrees[t] += 1
            graph.add_edge(str(new_node), str(t), alpha=alpha, beta=beta, power=power)
            if bidirectional:
                graph.add_edge(str(t), str(new_node), alpha=alpha, beta=beta, power=power)

    return graph


def build_braess_graph(
    *,
    include_shortcut: bool,
    linear: bool = True,
    demand_scale: float = 1.0,
) -> Tuple[CongestionGraph, str, str]:
    g = CongestionGraph()
    power = 1 if linear else 4
    variable_alpha = 1.0 / max(1.0, demand_scale) if linear else 1.0
    # Classic Braess-like setup
    g.add_edge("s", "a", alpha=variable_alpha, beta=0.0, power=power, name="s-a")
    g.add_edge("a", "t", alpha=0.0, beta=1.0, power=power, name="a-t")
    g.add_edge("s", "b", alpha=0.0, beta=1.0, power=power, name="s-b")
    g.add_edge("b", "t", alpha=variable_alpha, beta=0.0, power=power, name="b-t")
    if include_shortcut:
        g.add_edge("a", "b", alpha=0.0, beta=0.0, power=power, name="a-b-shortcut")
    return g, "s", "t"


def detect_braess_edges(
    graph: CongestionGraph,
    source: str,
    sink: str,
    n_agents: int,
    *,
    restarts: int = 3,
) -> List[EdgeKey]:
    baseline = worst_nash_equilibrium_cost(graph, source, sink, n_agents, restarts=restarts)
    paradoxical: List[EdgeKey] = []
    for edge_key in graph.edges().keys():
        reduced = graph.copy_without_edge(edge_key)
        try:
            reduced_cost = worst_nash_equilibrium_cost(
                reduced,
                source,
                sink,
                n_agents,
                restarts=restarts,
            )
        except ValueError:
            continue
        if reduced_cost < baseline:
            paradoxical.append(edge_key)
    return paradoxical


def pigovian_toll(edge: Edge, load: float) -> float:
    derivative = edge.alpha * edge.power * (load ** (edge.power - 1)) if edge.power > 0 else 0.0
    return load * derivative
