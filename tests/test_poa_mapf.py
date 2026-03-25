import unittest

from src.poa_mapf import (
    build_braess_graph,
    detect_braess_edges,
    frank_wolfe_equilibrium,
    min_cost_max_flow_social_optimum,
    price_of_anarchy,
    worst_nash_equilibrium_cost,
)


class TestPoAMAPF(unittest.TestCase):
    def test_social_optimum_not_worse_than_ne(self) -> None:
        g, s, t = build_braess_graph(include_shortcut=True, linear=True)
        n = 10
        ne = worst_nash_equilibrium_cost(g, s, t, n, restarts=2)
        so = min_cost_max_flow_social_optimum(g, s, t, n)
        self.assertGreaterEqual(ne, so)

    def test_price_of_anarchy_at_least_one(self) -> None:
        g, s, t = build_braess_graph(include_shortcut=True, linear=True)
        poa = price_of_anarchy(g, s, t, 10, restarts=2)
        self.assertGreaterEqual(poa, 1.0)

    def test_braess_shortcut_can_hurt(self) -> None:
        n = 40
        with_shortcut, s, t = build_braess_graph(
            include_shortcut=True,
            linear=True,
            demand_scale=float(n),
        )
        without_shortcut, _, _ = build_braess_graph(
            include_shortcut=False,
            linear=True,
            demand_scale=float(n),
        )

        cost_with = worst_nash_equilibrium_cost(with_shortcut, s, t, n, restarts=2)
        cost_without = worst_nash_equilibrium_cost(without_shortcut, s, t, n, restarts=2)
        self.assertGreaterEqual(cost_with, cost_without)

    def test_detects_paradoxical_shortcut_edge(self) -> None:
        n = 40
        g, s, t = build_braess_graph(
            include_shortcut=True,
            linear=True,
            demand_scale=float(n),
        )
        paradoxical = detect_braess_edges(g, s, t, n, restarts=2)
        names = {k[2] for k in paradoxical}
        self.assertIn("a-b-shortcut", names)

    def test_frank_wolfe_conserves_demand(self) -> None:
        g, s, t = build_braess_graph(include_shortcut=False, linear=False)
        demand = 3.0
        flow = frank_wolfe_equilibrium(g, s, t, demand, max_iters=50)
        outflow = sum(v for (u, _v, _n), v in flow.items() if u == s)
        self.assertAlmostEqual(outflow, demand, places=3)


if __name__ == "__main__":
    unittest.main()
