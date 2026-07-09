"""QIHC three-law theory utilities (division / refresh / trust)."""

from qihc.theory.higher_order import (
    HigherOrderInstance,
    generate_higher_order_instance,
    soft_effective_field,
    soft_mean_field_iterate,
    quadratize_higher_order,
    hubo_energy,
    quadratic_energy,
)
from qihc.theory.mean_field import (
    binary_entropy,
    contraction_lipschitz,
    free_energy_mean_field,
    mean_field_fixed_point,
    mean_field_iterate,
)
from qihc.theory.refresh import (
    estimate_tv_from_samples,
    fit_power_law,
    kl_bound_stale_field,
    residual_infeasibility_curve,
    tv_bound_stale_field,
)
from qihc.theory.trust import (
    estimate_trust_proxies,
    gated_lambda,
    optimal_lambda,
    snr_from_noise,
    usefulness_threshold_snr,
)

__all__ = [
    "HigherOrderInstance",
    "generate_higher_order_instance",
    "soft_effective_field",
    "soft_mean_field_iterate",
    "quadratize_higher_order",
    "hubo_energy",
    "quadratic_energy",
    "binary_entropy",
    "contraction_lipschitz",
    "free_energy_mean_field",
    "mean_field_fixed_point",
    "mean_field_iterate",
    "estimate_tv_from_samples",
    "fit_power_law",
    "kl_bound_stale_field",
    "residual_infeasibility_curve",
    "tv_bound_stale_field",
    "estimate_trust_proxies",
    "gated_lambda",
    "optimal_lambda",
    "snr_from_noise",
    "usefulness_threshold_snr",
]
