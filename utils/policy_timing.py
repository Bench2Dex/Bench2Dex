"""Policy-control timing helpers."""


def should_query_policy(sim_step: int, *, stride: int) -> bool:
    """Return True when a policy trained at capture rate should advance."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    return sim_step % stride == 0
