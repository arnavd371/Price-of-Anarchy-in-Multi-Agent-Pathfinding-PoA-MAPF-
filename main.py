from src.poa_mapf import (
    build_braess_graph,
    detect_braess_edges,
    min_cost_max_flow_social_optimum,
    price_of_anarchy,
    worst_nash_equilibrium_cost,
)


def run_demo() -> None:
    agents = 40
    g_with, s, t = build_braess_graph(include_shortcut=True, linear=True, demand_scale=float(agents))
    g_without, _, _ = build_braess_graph(include_shortcut=False, linear=True, demand_scale=float(agents))

    ne_with = worst_nash_equilibrium_cost(g_with, s, t, agents)
    ne_without = worst_nash_equilibrium_cost(g_without, s, t, agents)
    so_with = min_cost_max_flow_social_optimum(g_with, s, t, agents)
    poa_with = price_of_anarchy(g_with, s, t, agents)

    print(f"Agents: {agents}")
    print(f"NE social cost (with shortcut): {ne_with:.3f}")
    print(f"NE social cost (without shortcut): {ne_without:.3f}")
    print(f"Social optimum (with shortcut): {so_with:.3f}")
    print(f"PoA (with shortcut): {poa_with:.3f}")
    print("Paradoxical edges:", detect_braess_edges(g_with, s, t, agents))


if __name__ == "__main__":
    run_demo()
