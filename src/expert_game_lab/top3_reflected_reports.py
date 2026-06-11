from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _experiments_impl():
    from . import experiments_impl as _impl

    return _impl


@dataclass(frozen=True)
class K5Top3ReflectedBoundaryConvolutionResidualRow:
    a: int
    max_abs_residual: float
    arg_n: int | None
    residual_at_arg: float


@dataclass(frozen=True)
class K5Top3ReflectedBoundaryConvolutionScanRow:
    name: str
    min_value: float
    max_scaled: float
    argmax_n: int | None
    argmax_a: int | None
    argmax_value: float
    sign_holds: bool


@dataclass(frozen=True)
class K5Top3ReflectedBoundaryConvolutionBinRow:
    label: str
    count: int
    max_scaled: float
    argmax_n: int | None
    argmax_a: int | None
    argmax_ratio: float
    sign_holds: bool


@dataclass(frozen=True)
class K5Top3ReflectedBoundaryKernelEstimateRow:
    heat_c: float
    kernel_constant: float


@dataclass(frozen=True)
class K5Top3ReflectedBoundaryEnvelopeRow:
    source_name: str
    source_constant: float
    max_scaled_upper: float
    argmax_n: int | None
    argmax_a: int | None


@dataclass(frozen=True)
class K5Top3ReflectedBoundaryPointwiseRow:
    max_n2_summand: float
    argmax_n: int | None
    argmax_a: int | None
    argmax_t: int | None
    argmax_value: float


@dataclass(frozen=True)
class K5Top3ReflectedConvolutionAnalyticScanRow:
    heat_c: float
    kernel_constant: float
    source_constant: float
    unit_max_scaled: float
    max_scaled_upper: float
    argmax_n: int | None
    argmax_a: int | None


@dataclass(frozen=True)
class K5Top3ReflectedConvolutionRegimeRow:
    label: str
    max_scaled_upper: float
    argmax_n: int | None
    argmax_a: int | None


@dataclass(frozen=True)
class K5Top3ReflectedSourceSummaryRow:
    max_value: float
    argmax_value_t: int | None
    argmax_value_a: int | None
    max_t_scaled: float
    argmax_t_scaled_t: int | None
    argmax_t_scaled_a: int | None
    max_t_over_a_scaled: float
    argmax_t_over_a_scaled_t: int | None
    argmax_t_over_a_scaled_a: int | None


@dataclass(frozen=True)
class K5Top3ReflectedSourceBinRow:
    label: str
    count: int
    max_value: float
    max_t_scaled: float
    argmax_t: int | None
    argmax_a: int | None
    argmax_ratio: float


@dataclass(frozen=True)
class K5Top3ReflectedSourceGaussianEnvelopeRow:
    family_name: str
    heat_c: float
    source_constant: float
    argmax_t: int | None
    argmax_a: int | None
    argmax_ratio: float
    hits_t_boundary: bool
    hits_a_boundary: bool


@dataclass(frozen=True)
class K5Top3ReflectedConvolutionGaussianScanRow:
    kernel_heat_c: float
    kernel_constant: float
    source_family_name: str
    source_heat_c: float
    source_constant: float
    unit_max_scaled: float
    max_scaled_upper: float
    argmax_n: int | None
    argmax_a: int | None


@dataclass(frozen=True)
class K5Top3ReflectedConvolutionComparisonRow:
    name: str
    max_scaled: float
    argmax_n: int | None
    argmax_a: int | None
    argmax_value: float


@dataclass(frozen=True)
class K5Top3ReflectedGradientConvolutionRegimeRow:
    label: str
    max_scaled_upper: float
    argmax_n: int | None
    argmax_a: int | None


def k5_top3_reflected_boundary_convolution_residual_rows(
    max_n: int = 120,
    dense_a_limit: int = 60,
) -> tuple[K5Top3ReflectedBoundaryConvolutionResidualRow, ...]:
    _impl = _experiments_impl()
    rows: list[K5Top3ReflectedBoundaryConvolutionResidualRow] = []
    max_a = min(max_n + 2, dense_a_limit)
    for a in range(1, max_a + 1):
        initial_series, root_series, boundary_series, _ = _impl._top3_diagonal_e1_representation_series(max_n, a)
        max_abs_residual = 0.0
        arg_n: int | None = None
        residual_at_arg = 0.0
        for n_value in range(max_n + 1):
            direct = _impl._top3_diagonal_discrepancy_value(1, n_value, a)
            residual = direct - float(initial_series[n_value] + root_series[n_value] + boundary_series[n_value])
            if arg_n is None:
                arg_n = n_value
            if abs(residual) > max_abs_residual:
                max_abs_residual = abs(residual)
                arg_n = n_value
                residual_at_arg = residual
        rows.append(
            K5Top3ReflectedBoundaryConvolutionResidualRow(
                a=a,
                max_abs_residual=max_abs_residual,
                arg_n=arg_n,
                residual_at_arg=residual_at_arg,
            )
        )
    return tuple(rows)


def k5_top3_reflected_boundary_convolution_scan_rows(
    max_n: int = 500,
    dense_a_limit: int = 200,
) -> tuple[K5Top3ReflectedBoundaryConvolutionScanRow, ...]:
    _impl = _experiments_impl()
    stats = {
        name: {
            "min_value": float("inf"),
            "max_scaled": 0.0,
            "argmax_n": None,
            "argmax_a": None,
            "argmax_value": 0.0,
            "sign_holds": True,
        }
        for name in ("Rref", "root", "initial", "E1")
    }
    for a in range(1, dense_a_limit + 1):
        initial_series, root_series, boundary_series, total_series = _impl._top3_diagonal_e1_representation_series(max_n, a)
        start_n = max(0, a - 2)
        for n_value in range(start_n, max_n + 1):
            values = {
                "Rref": float(boundary_series[n_value]),
                "root": float(root_series[n_value]),
                "initial": float(initial_series[n_value]),
                "E1": float(total_series[n_value]),
            }
            for name, value in values.items():
                stat = stats[name]
                stat["min_value"] = min(stat["min_value"], value)
                if value < -1e-12:
                    stat["sign_holds"] = False
                scaled = (n_value + 1) * max(value, 0.0)
                if scaled > stat["max_scaled"]:
                    stat["max_scaled"] = scaled
                    stat["argmax_n"] = n_value
                    stat["argmax_a"] = a
                    stat["argmax_value"] = value
    rows = []
    for name in ("Rref", "root", "initial", "E1"):
        stat = stats[name]
        rows.append(
            K5Top3ReflectedBoundaryConvolutionScanRow(
                name=name,
                min_value=0.0 if stat["min_value"] == float("inf") else float(stat["min_value"]),
                max_scaled=float(stat["max_scaled"]),
                argmax_n=stat["argmax_n"],
                argmax_a=stat["argmax_a"],
                argmax_value=float(stat["argmax_value"]),
                sign_holds=bool(stat["sign_holds"]),
            )
        )
    return tuple(rows)


def k5_top3_reflected_boundary_convolution_bin_rows(
    max_n: int = 500,
    dense_a_limit: int = 200,
) -> tuple[K5Top3ReflectedBoundaryConvolutionBinRow, ...]:
    _impl = _experiments_impl()
    bin_specs = (
        ("[0,0.5)", 0.0, 0.5),
        ("[0.5,1)", 0.5, 1.0),
        ("[1,2)", 1.0, 2.0),
        ("[2,4)", 2.0, 4.0),
        ("[4,inf)", 4.0, None),
    )
    stats = {
        label: {
            "count": 0,
            "max_scaled": 0.0,
            "argmax_n": None,
            "argmax_a": None,
            "argmax_ratio": 0.0,
            "sign_holds": True,
        }
        for label, _, _ in bin_specs
    }
    for a in range(1, dense_a_limit + 1):
        _, _, boundary_series, _ = _impl._top3_diagonal_e1_representation_series(max_n, a)
        start_n = max(0, a - 2)
        for n_value in range(start_n, max_n + 1):
            ratio = a / float((n_value + 1) ** 0.5)
            for label, lower, upper in bin_specs:
                if ratio < lower:
                    continue
                if upper is not None and ratio >= upper:
                    continue
                stat = stats[label]
                value = float(boundary_series[n_value])
                stat["count"] += 1
                if value < -1e-12:
                    stat["sign_holds"] = False
                scaled = (n_value + 1) * max(value, 0.0)
                if scaled > stat["max_scaled"]:
                    stat["max_scaled"] = scaled
                    stat["argmax_n"] = n_value
                    stat["argmax_a"] = a
                    stat["argmax_ratio"] = ratio
                break
    return tuple(
        K5Top3ReflectedBoundaryConvolutionBinRow(
            label=label,
            count=int(stats[label]["count"]),
            max_scaled=float(stats[label]["max_scaled"]),
            argmax_n=stats[label]["argmax_n"],
            argmax_a=stats[label]["argmax_a"],
            argmax_ratio=float(stats[label]["argmax_ratio"]),
            sign_holds=bool(stats[label]["sign_holds"]),
        )
        for label, _, _ in bin_specs
    )


def k5_top3_reflected_boundary_kernel_estimate_rows(
    max_n: int = 500,
    dense_a_limit: int = 200,
    heat_c_values: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0),
) -> tuple[K5Top3ReflectedBoundaryKernelEstimateRow, ...]:
    _impl = _experiments_impl()
    rows: list[K5Top3ReflectedBoundaryKernelEstimateRow] = []
    for heat_c in heat_c_values:
        kernel_constant = 0.0
        for a in range(1, dense_a_limit + 1):
            kernel = _impl._top3_diagonal_dirichlet_heat_kernel(a, max_n - 1)
            for m in range(max_n):
                base = a * float(np.exp(-(a**2) / (heat_c * (m + 1)))) / ((m + 1) ** 1.5)
                if base <= 0.0:
                    continue
                kernel_constant = max(kernel_constant, float(kernel[m, 0, a - 1]) / base)
        rows.append(
            K5Top3ReflectedBoundaryKernelEstimateRow(
                heat_c=heat_c,
                kernel_constant=kernel_constant,
            )
        )
    return tuple(rows)


def k5_top3_reflected_boundary_pointwise_row(
    max_n: int = 500,
    dense_a_limit: int = 200,
) -> K5Top3ReflectedBoundaryPointwiseRow:
    _impl = _experiments_impl()
    max_n2_summand = 0.0
    argmax_n: int | None = None
    argmax_a: int | None = None
    argmax_t: int | None = None
    argmax_value = 0.0
    for a in range(1, dense_a_limit + 1):
        kernel = _impl._top3_diagonal_dirichlet_heat_kernel(a, max_n - 1)
        boundary_source = np.array([_impl._top3_diagonal_reflected_boundary_child_value(t, a) for t in range(max_n)], dtype=float)
        start_n = max(1, a - 2)
        for n_value in range(start_n, max_n + 1):
            weights = 0.5 * kernel[:n_value, 0, a - 1][::-1] * boundary_source[:n_value]
            if weights.size == 0:
                continue
            t_index = int(np.argmax(weights))
            value = float(weights[t_index])
            scaled = ((n_value + 1) ** 2) * value
            if scaled > max_n2_summand:
                max_n2_summand = scaled
                argmax_n = n_value
                argmax_a = a
                argmax_t = t_index
                argmax_value = value
    return K5Top3ReflectedBoundaryPointwiseRow(
        max_n2_summand=max_n2_summand,
        argmax_n=argmax_n,
        argmax_a=argmax_a,
        argmax_t=argmax_t,
        argmax_value=argmax_value,
    )


def k5_top3_reflected_boundary_envelope_rows(
    max_n: int = 500,
    dense_a_limit: int = 200,
) -> tuple[K5Top3ReflectedBoundaryEnvelopeRow, ...]:
    _impl = _experiments_impl()
    rows: list[K5Top3ReflectedBoundaryEnvelopeRow] = []
    for source_name, envelope_fn in _top3_reflected_boundary_envelope_specs():
        source_constant = 0.0
        for t in range(max_n):
            for a in range(1, dense_a_limit + 1):
                envelope_value = envelope_fn(t, a)
                if envelope_value <= 0.0:
                    continue
                source_constant = max(
                    source_constant,
                    _impl._top3_diagonal_reflected_boundary_child_value(t, a) / envelope_value,
                )
        max_scaled_upper = 0.0
        argmax_n: int | None = None
        argmax_a: int | None = None
        for a in range(1, dense_a_limit + 1):
            kernel = _impl._top3_diagonal_dirichlet_heat_kernel(a, max_n - 1)
            envelope_values = np.array([envelope_fn(t, a) for t in range(max_n)], dtype=float)
            start_n = max(1, a - 2)
            for n_value in range(start_n, max_n + 1):
                upper = 0.5 * source_constant * float(kernel[:n_value, 0, a - 1][::-1] @ envelope_values[:n_value])
                scaled = (n_value + 1) * upper
                if scaled > max_scaled_upper:
                    max_scaled_upper = scaled
                    argmax_n = n_value
                    argmax_a = a
        rows.append(
            K5Top3ReflectedBoundaryEnvelopeRow(
                source_name=source_name,
                source_constant=source_constant,
                max_scaled_upper=max_scaled_upper,
                argmax_n=argmax_n,
                argmax_a=argmax_a,
            )
        )
    return tuple(rows)


def _top3_reflected_boundary_envelope_specs() -> tuple[tuple[str, Callable[[int, int], float]], ...]:
    return (
        ("min(1,(a+1)/(t+1))", lambda t, a: min(1.0, (a + 1) / (t + 1))),
        ("(a+1)/(t+1)", lambda t, a: (a + 1) / (t + 1)),
    )


def _top3_reflected_source_bin_specs() -> tuple[tuple[str, float, float | None], ...]:
    return (
        ("[0,0.5)", 0.0, 0.5),
        ("[0.5,1)", 0.5, 1.0),
        ("[1,2)", 1.0, 2.0),
        ("[2,4)", 2.0, 4.0),
        ("[4,inf)", 4.0, None),
    )


def _top3_reflected_source_gaussian_family_specs() -> tuple[str, ...]:
    return (
        "min(1,(a+1)/(t+1))*exp",
        "(a+1)/(t+1)*exp",
        "(a+1)/(t+1)^(3/2)*exp",
        "(a+1)^2/(t+1)^(3/2)*exp",
    )


def _top3_reflected_source_gaussian_envelope_value(
    t: int,
    a: int,
    family_name: str,
    heat_c: float,
) -> float:
    decay = float(np.exp(-(a**2) / (heat_c * (t + 1))))
    if family_name == "min(1,(a+1)/(t+1))*exp":
        return min(1.0, (a + 1) / (t + 1)) * decay
    if family_name == "(a+1)/(t+1)*exp":
        return ((a + 1) / (t + 1)) * decay
    if family_name == "(a+1)/(t+1)^(3/2)*exp":
        return ((a + 1) / ((t + 1) ** 1.5)) * decay
    if family_name == "(a+1)^2/(t+1)^(3/2)*exp":
        return (((a + 1) ** 2) / ((t + 1) ** 1.5)) * decay
    raise ValueError(f"unknown Gaussian source family {family_name!r}")


def _top3_reflected_source_gaussian_profile(
    a: int,
    max_n: int,
    family_name: str,
    heat_c: float,
) -> np.ndarray:
    if max_n <= 0:
        return np.zeros(0, dtype=float)
    t_values = np.arange(max_n, dtype=float)
    decay = np.exp(-(a**2) / (heat_c * (t_values + 1.0)))
    if family_name == "min(1,(a+1)/(t+1))*exp":
        prefactor = np.minimum(1.0, (a + 1) / (t_values + 1.0))
    elif family_name == "(a+1)/(t+1)*exp":
        prefactor = (a + 1) / (t_values + 1.0)
    elif family_name == "(a+1)/(t+1)^(3/2)*exp":
        prefactor = (a + 1) / np.power(t_values + 1.0, 1.5)
    elif family_name == "(a+1)^2/(t+1)^(3/2)*exp":
        prefactor = ((a + 1) ** 2) / np.power(t_values + 1.0, 1.5)
    else:
        raise ValueError(f"unknown Gaussian source family {family_name!r}")
    return prefactor * decay


def _top3_reflected_source_grid(
    max_n: int,
    dense_a_limit: int,
) -> np.ndarray:
    _impl = _experiments_impl()
    if max_n < 0:
        raise ValueError(f"max_n must be nonnegative, got {max_n}")
    if dense_a_limit < 1:
        raise ValueError(f"dense_a_limit must be positive, got {dense_a_limit}")
    grid = np.zeros((max_n + 1, dense_a_limit + 1), dtype=float)
    for t in range(max_n + 1):
        max_a = min(t + 2, dense_a_limit)
        for a in range(1, max_a + 1):
            grid[t, a] = _impl._top3_diagonal_reflected_boundary_child_value(t, a)
    return grid


def k5_top3_reflected_source_summary_row(
    max_n: int = 5000,
    dense_a_limit: int = 1000,
    source_grid: np.ndarray | None = None,
) -> K5Top3ReflectedSourceSummaryRow:
    grid = source_grid if source_grid is not None else _top3_reflected_source_grid(max_n=max_n, dense_a_limit=dense_a_limit)
    max_value = 0.0
    argmax_value_t: int | None = None
    argmax_value_a: int | None = None
    max_t_scaled = 0.0
    argmax_t_scaled_t: int | None = None
    argmax_t_scaled_a: int | None = None
    max_t_over_a_scaled = 0.0
    argmax_t_over_a_scaled_t: int | None = None
    argmax_t_over_a_scaled_a: int | None = None
    for t in range(max_n + 1):
        max_a = min(t + 2, dense_a_limit)
        if max_a < 1:
            continue
        a_values = np.arange(1, max_a + 1, dtype=int)
        values = np.maximum(grid[t, a_values], 0.0)
        if values.size == 0:
            continue
        local_index = int(np.argmax(values))
        local_value = float(values[local_index])
        if local_value > max_value:
            max_value = local_value
            argmax_value_t = t
            argmax_value_a = int(a_values[local_index])
        scaled_t = (t + 1) * values
        local_scaled_index = int(np.argmax(scaled_t))
        local_scaled_value = float(scaled_t[local_scaled_index])
        if local_scaled_value > max_t_scaled:
            max_t_scaled = local_scaled_value
            argmax_t_scaled_t = t
            argmax_t_scaled_a = int(a_values[local_scaled_index])
        scaled_t_over_a = scaled_t / (a_values + 1)
        local_ratio_index = int(np.argmax(scaled_t_over_a))
        local_ratio_value = float(scaled_t_over_a[local_ratio_index])
        if local_ratio_value > max_t_over_a_scaled:
            max_t_over_a_scaled = local_ratio_value
            argmax_t_over_a_scaled_t = t
            argmax_t_over_a_scaled_a = int(a_values[local_ratio_index])
    return K5Top3ReflectedSourceSummaryRow(
        max_value=max_value,
        argmax_value_t=argmax_value_t,
        argmax_value_a=argmax_value_a,
        max_t_scaled=max_t_scaled,
        argmax_t_scaled_t=argmax_t_scaled_t,
        argmax_t_scaled_a=argmax_t_scaled_a,
        max_t_over_a_scaled=max_t_over_a_scaled,
        argmax_t_over_a_scaled_t=argmax_t_over_a_scaled_t,
        argmax_t_over_a_scaled_a=argmax_t_over_a_scaled_a,
    )


def k5_top3_reflected_source_bin_rows(
    max_n: int = 5000,
    dense_a_limit: int = 1000,
    source_grid: np.ndarray | None = None,
) -> tuple[K5Top3ReflectedSourceBinRow, ...]:
    grid = source_grid if source_grid is not None else _top3_reflected_source_grid(max_n=max_n, dense_a_limit=dense_a_limit)
    bin_specs = _top3_reflected_source_bin_specs()
    stats = {
        label: {
            "count": 0,
            "max_value": 0.0,
            "max_t_scaled": 0.0,
            "argmax_t": None,
            "argmax_a": None,
            "argmax_ratio": 0.0,
        }
        for label, _, _ in bin_specs
    }
    for t in range(max_n + 1):
        max_a = min(t + 2, dense_a_limit)
        if max_a < 1:
            continue
        scale = float((t + 1) ** 0.5)
        for a in range(1, max_a + 1):
            ratio = a / scale
            value = max(float(grid[t, a]), 0.0)
            for label, lower, upper in bin_specs:
                if ratio < lower:
                    continue
                if upper is not None and ratio >= upper:
                    continue
                stat = stats[label]
                stat["count"] += 1
                if value > stat["max_value"]:
                    stat["max_value"] = value
                scaled = (t + 1) * value
                if scaled > stat["max_t_scaled"]:
                    stat["max_t_scaled"] = scaled
                    stat["argmax_t"] = t
                    stat["argmax_a"] = a
                    stat["argmax_ratio"] = ratio
                break
    return tuple(
        K5Top3ReflectedSourceBinRow(
            label=label,
            count=int(stats[label]["count"]),
            max_value=float(stats[label]["max_value"]),
            max_t_scaled=float(stats[label]["max_t_scaled"]),
            argmax_t=stats[label]["argmax_t"],
            argmax_a=stats[label]["argmax_a"],
            argmax_ratio=float(stats[label]["argmax_ratio"]),
        )
        for label, _, _ in bin_specs
    )


def k5_top3_reflected_source_gaussian_envelope_rows(
    max_n: int = 5000,
    dense_a_limit: int = 1000,
    heat_c_values: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0),
    source_grid: np.ndarray | None = None,
) -> tuple[K5Top3ReflectedSourceGaussianEnvelopeRow, ...]:
    grid = source_grid if source_grid is not None else _top3_reflected_source_grid(max_n=max_n, dense_a_limit=dense_a_limit)
    rows: list[K5Top3ReflectedSourceGaussianEnvelopeRow] = []
    for family_name in _top3_reflected_source_gaussian_family_specs():
        for heat_c in heat_c_values:
            source_constant = 0.0
            argmax_t: int | None = None
            argmax_a: int | None = None
            argmax_ratio = 0.0
            for t in range(max_n + 1):
                max_a = min(t + 2, dense_a_limit)
                if max_a < 1:
                    continue
                a_values = np.arange(1, max_a + 1, dtype=int)
                values = np.maximum(grid[t, a_values], 0.0)
                base = np.array(
                    [_top3_reflected_source_gaussian_envelope_value(t, int(a), family_name, heat_c) for a in a_values],
                    dtype=float,
                )
                ratios = np.zeros_like(base)
                positive_mask = base > 0.0
                ratios[positive_mask] = values[positive_mask] / base[positive_mask]
                ratios[np.logical_not(positive_mask) & (values > 0.0)] = float("inf")
                local_index = int(np.argmax(ratios)) if ratios.size else 0
                local_value = float(ratios[local_index]) if ratios.size else 0.0
                if local_value > source_constant:
                    source_constant = local_value
                    argmax_t = t
                    argmax_a = int(a_values[local_index])
                    argmax_ratio = argmax_a / float((t + 1) ** 0.5)
            rows.append(
                K5Top3ReflectedSourceGaussianEnvelopeRow(
                    family_name=family_name,
                    heat_c=heat_c,
                    source_constant=source_constant,
                    argmax_t=argmax_t,
                    argmax_a=argmax_a,
                    argmax_ratio=argmax_ratio,
                    hits_t_boundary=argmax_t == max_n,
                    hits_a_boundary=bool(argmax_t is not None and argmax_a == min(argmax_t + 2, dense_a_limit)),
                )
            )
    return tuple(rows)


def _top3_choose_reflected_source_gaussian_envelope(
    rows: tuple[K5Top3ReflectedSourceGaussianEnvelopeRow, ...],
) -> K5Top3ReflectedSourceGaussianEnvelopeRow | None:
    finite_rows = [row for row in rows if np.isfinite(row.source_constant)]
    if not finite_rows:
        return None
    preferred_rows = [
        row
        for row in finite_rows
        if not row.hits_t_boundary and not row.hits_a_boundary and 0.5 <= row.argmax_ratio <= 2.5
    ]
    if preferred_rows:
        return min(preferred_rows, key=lambda item: (item.source_constant, abs(item.argmax_ratio - 1.0)))
    interior_rows = [row for row in finite_rows if not row.hits_t_boundary and not row.hits_a_boundary]
    if interior_rows:
        return min(interior_rows, key=lambda item: (item.source_constant, abs(item.argmax_ratio - 1.0)))
    return min(finite_rows, key=lambda item: (item.source_constant, item.hits_t_boundary, item.hits_a_boundary))


def _top3_choose_reflected_source_gradient_envelope(
    rows: tuple[K5Top3ReflectedSourceGaussianEnvelopeRow, ...],
) -> K5Top3ReflectedSourceGaussianEnvelopeRow | None:
    gradient_rows = tuple(
        row
        for row in rows
        if row.family_name == "(a+1)/(t+1)^(3/2)*exp" and np.isfinite(row.source_constant)
    )
    if not gradient_rows:
        return None
    interior_rows = [row for row in gradient_rows if not row.hits_t_boundary and not row.hits_a_boundary]
    if interior_rows:
        return min(interior_rows, key=lambda item: (item.source_constant, abs(item.argmax_ratio - 1.0)))
    return min(gradient_rows, key=lambda item: (item.source_constant, item.hits_t_boundary, item.hits_a_boundary))


def _top3_reflected_convolution_fft(values_left: np.ndarray, values_right: np.ndarray, output_len: int) -> np.ndarray:
    convolution_len = len(values_left) + len(values_right) - 1
    fft_len = 1 << max(0, convolution_len - 1).bit_length()
    transformed = np.fft.rfft(values_left, n=fft_len) * np.fft.rfft(values_right, n=fft_len)
    convolved = np.fft.irfft(transformed, n=fft_len)[:output_len]
    convolved[np.abs(convolved) < 1e-15] = 0.0
    return convolved


def _top3_reflected_convolution_heat_profile(
    a: int,
    max_n: int,
    heat_c: float,
) -> np.ndarray:
    profile = np.zeros(max_n + 1, dtype=float)
    if max_n <= 0:
        return profile
    m_values = np.arange(1, max_n + 1, dtype=float)
    profile[1:] = a * np.exp(-(a**2) / (heat_c * m_values)) / np.power(m_values, 1.5)
    return profile


def _top3_reflected_convolution_source_profile(
    a: int,
    max_n: int,
    source_name: str = "min(1,(a+1)/(t+1))",
) -> np.ndarray:
    if max_n <= 0:
        return np.zeros(0, dtype=float)
    t_values = np.arange(max_n, dtype=float)
    if source_name == "min(1,(a+1)/(t+1))":
        return np.minimum(1.0, (a + 1) / (t_values + 1.0))
    if source_name == "(a+1)/(t+1)":
        return (a + 1) / (t_values + 1.0)
    raise ValueError(f"unknown source envelope {source_name!r}")


def k5_top3_reflected_convolution_gaussian_scan_rows(
    max_n: int = 5000,
    dense_a_limit: int = 1000,
    source_row: K5Top3ReflectedSourceGaussianEnvelopeRow | None = None,
    verification_max_n: int = 120,
    verification_a_limit: int = 60,
    kernel_heat_c_values: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0),
) -> tuple[K5Top3ReflectedConvolutionGaussianScanRow, ...]:
    if source_row is None:
        source_row = _top3_choose_reflected_source_gaussian_envelope(
            k5_top3_reflected_source_gaussian_envelope_rows(
                max_n=verification_max_n,
                dense_a_limit=verification_a_limit,
            )
        )
    if source_row is None:
        return ()
    kernel_rows = k5_top3_reflected_boundary_kernel_estimate_rows(
        max_n=verification_max_n,
        dense_a_limit=verification_a_limit,
        heat_c_values=kernel_heat_c_values,
    )
    n_weights = np.arange(max_n + 1, dtype=float) + 1.0
    rows: list[K5Top3ReflectedConvolutionGaussianScanRow] = []
    for kernel_row in kernel_rows:
        unit_max_scaled = 0.0
        argmax_n: int | None = None
        argmax_a: int | None = None
        for a in range(1, dense_a_limit + 1):
            heat_profile = _top3_reflected_convolution_heat_profile(a=a, max_n=max_n, heat_c=kernel_row.heat_c)
            source_profile = _top3_reflected_source_gaussian_profile(
                a=a,
                max_n=max_n,
                family_name=source_row.family_name,
                heat_c=source_row.heat_c,
            )
            convolved = _top3_reflected_convolution_fft(heat_profile, source_profile, max_n + 1)
            scaled = 0.5 * n_weights * np.maximum(convolved, 0.0)
            start_n = max(0, a - 2)
            local_index = int(np.argmax(scaled[start_n:])) if start_n <= max_n else 0
            n_value = start_n + local_index
            value = float(scaled[n_value])
            if value > unit_max_scaled:
                unit_max_scaled = value
                argmax_n = n_value
                argmax_a = a
        rows.append(
            K5Top3ReflectedConvolutionGaussianScanRow(
                kernel_heat_c=kernel_row.heat_c,
                kernel_constant=kernel_row.kernel_constant,
                source_family_name=source_row.family_name,
                source_heat_c=source_row.heat_c,
                source_constant=source_row.source_constant,
                unit_max_scaled=unit_max_scaled,
                max_scaled_upper=kernel_row.kernel_constant * source_row.source_constant * unit_max_scaled,
                argmax_n=argmax_n,
                argmax_a=argmax_a,
            )
        )
    return tuple(rows)


def k5_top3_reflected_gradient_convolution_scan_rows(
    max_n: int = 5000,
    dense_a_limit: int = 1000,
    source_rows: tuple[K5Top3ReflectedSourceGaussianEnvelopeRow, ...] | None = None,
    verification_max_n: int = 120,
    verification_a_limit: int = 60,
    kernel_heat_c_values: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0),
    source_heat_c_values: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0),
) -> tuple[K5Top3ReflectedConvolutionGaussianScanRow, ...]:
    if source_rows is None:
        source_rows = k5_top3_reflected_source_gaussian_envelope_rows(
            max_n=max_n,
            dense_a_limit=dense_a_limit,
            heat_c_values=source_heat_c_values,
        )
    source_by_c = {
        row.heat_c: row
        for row in source_rows
        if row.family_name == "(a+1)/(t+1)^(3/2)*exp" and row.heat_c in source_heat_c_values
    }
    kernel_rows = k5_top3_reflected_boundary_kernel_estimate_rows(
        max_n=verification_max_n,
        dense_a_limit=verification_a_limit,
        heat_c_values=kernel_heat_c_values,
    )
    n_weights = np.arange(max_n + 1, dtype=float) + 1.0
    rows: list[K5Top3ReflectedConvolutionGaussianScanRow] = []
    for kernel_row in kernel_rows:
        for source_heat_c in source_heat_c_values:
            source_row = source_by_c.get(source_heat_c)
            if source_row is None:
                continue
            unit_max_scaled = 0.0
            argmax_n: int | None = None
            argmax_a: int | None = None
            for a in range(1, dense_a_limit + 1):
                heat_profile = _top3_reflected_convolution_heat_profile(a=a, max_n=max_n, heat_c=kernel_row.heat_c)
                source_profile = _top3_reflected_source_gaussian_profile(
                    a=a,
                    max_n=max_n,
                    family_name=source_row.family_name,
                    heat_c=source_row.heat_c,
                )
                convolved = _top3_reflected_convolution_fft(heat_profile, source_profile, max_n + 1)
                scaled = 0.5 * n_weights * np.maximum(convolved, 0.0)
                start_n = max(0, a - 2)
                local_index = int(np.argmax(scaled[start_n:])) if start_n <= max_n else 0
                n_value = start_n + local_index
                value = float(scaled[n_value])
                if value > unit_max_scaled:
                    unit_max_scaled = value
                    argmax_n = n_value
                    argmax_a = a
            rows.append(
                K5Top3ReflectedConvolutionGaussianScanRow(
                    kernel_heat_c=kernel_row.heat_c,
                    kernel_constant=kernel_row.kernel_constant,
                    source_family_name=source_row.family_name,
                    source_heat_c=source_row.heat_c,
                    source_constant=source_row.source_constant,
                    unit_max_scaled=unit_max_scaled,
                    max_scaled_upper=kernel_row.kernel_constant * source_row.source_constant * unit_max_scaled,
                    argmax_n=argmax_n,
                    argmax_a=argmax_a,
                )
            )

    if max_n <= 200 and dense_a_limit <= 100:
        best_source_row = _top3_choose_reflected_source_gradient_envelope(source_rows)
        comparison_rows = k5_top3_reflected_gradient_convolution_comparison_rows(
            max_n=max_n,
            dense_a_limit=dense_a_limit,
            source_row=best_source_row,
        )
        exact_row = next((row for row in comparison_rows if row.name == "exact Rref"), None)
        if exact_row is not None:
            rows.append(
                K5Top3ReflectedConvolutionGaussianScanRow(
                    kernel_heat_c=0.0,
                    kernel_constant=1.0,
                    source_family_name="exact-reflected-source",
                    source_heat_c=0.0,
                    source_constant=1.0,
                    unit_max_scaled=exact_row.max_scaled,
                    max_scaled_upper=exact_row.max_scaled,
                    argmax_n=exact_row.argmax_n,
                    argmax_a=exact_row.argmax_a,
                )
            )
    return tuple(rows)


def k5_top3_reflected_convolution_exact_upper_comparison_rows(
    max_n: int = 120,
    dense_a_limit: int = 60,
    source_row: K5Top3ReflectedSourceGaussianEnvelopeRow | None = None,
) -> tuple[K5Top3ReflectedConvolutionComparisonRow, ...]:
    _impl = _experiments_impl()
    if max_n < 1:
        return (
            K5Top3ReflectedConvolutionComparisonRow("exact Rref", 0.0, None, None, 0.0),
            K5Top3ReflectedConvolutionComparisonRow("old upper", 0.0, None, None, 0.0),
            K5Top3ReflectedConvolutionComparisonRow("new Gaussian upper", 0.0, None, None, 0.0),
        )
    if source_row is None:
        source_row = _top3_choose_reflected_source_gaussian_envelope(
            k5_top3_reflected_source_gaussian_envelope_rows(
                max_n=max_n,
                dense_a_limit=dense_a_limit,
            )
        )
    if source_row is None:
        return ()
    old_rows = {
        row.source_name: row
        for row in k5_top3_reflected_boundary_envelope_rows(
            max_n=max_n,
            dense_a_limit=dense_a_limit,
        )
    }
    old_constant = old_rows["min(1,(a+1)/(t+1))"].source_constant
    source_grid = _top3_reflected_source_grid(max_n=max_n - 1, dense_a_limit=dense_a_limit)
    stats = {
        "exact Rref": {"max_scaled": 0.0, "argmax_n": None, "argmax_a": None, "argmax_value": 0.0},
        "old upper": {"max_scaled": 0.0, "argmax_n": None, "argmax_a": None, "argmax_value": 0.0},
        "new Gaussian upper": {"max_scaled": 0.0, "argmax_n": None, "argmax_a": None, "argmax_value": 0.0},
    }
    for a in range(1, dense_a_limit + 1):
        kernel = _impl._top3_diagonal_dirichlet_heat_kernel(a, max_n - 1)
        exact_source = np.maximum(source_grid[:max_n, a], 0.0)
        old_source = old_constant * _top3_reflected_convolution_source_profile(a=a, max_n=max_n, source_name="min(1,(a+1)/(t+1))")
        new_source = source_row.source_constant * _top3_reflected_source_gaussian_profile(
            a=a,
            max_n=max_n,
            family_name=source_row.family_name,
            heat_c=source_row.heat_c,
        )
        start_n = max(1, a - 2)
        for n_value in range(start_n, max_n + 1):
            kernel_slice = kernel[:n_value, 0, a - 1][::-1]
            exact_value = 0.5 * float(kernel_slice @ exact_source[:n_value])
            old_upper = 0.5 * float(kernel_slice @ old_source[:n_value])
            new_upper = 0.5 * float(kernel_slice @ new_source[:n_value])
            for name, value in (
                ("exact Rref", exact_value),
                ("old upper", old_upper),
                ("new Gaussian upper", new_upper),
            ):
                scaled = (n_value + 1) * max(value, 0.0)
                if scaled > stats[name]["max_scaled"]:
                    stats[name]["max_scaled"] = scaled
                    stats[name]["argmax_n"] = n_value
                    stats[name]["argmax_a"] = a
                    stats[name]["argmax_value"] = value
    return tuple(
        K5Top3ReflectedConvolutionComparisonRow(
            name=name,
            max_scaled=float(stat["max_scaled"]),
            argmax_n=stat["argmax_n"],
            argmax_a=stat["argmax_a"],
            argmax_value=float(stat["argmax_value"]),
        )
        for name, stat in stats.items()
    )


def k5_top3_reflected_gradient_convolution_comparison_rows(
    max_n: int = 120,
    dense_a_limit: int = 60,
    source_row: K5Top3ReflectedSourceGaussianEnvelopeRow | None = None,
) -> tuple[K5Top3ReflectedConvolutionComparisonRow, ...]:
    _impl = _experiments_impl()
    if max_n < 1:
        return (
            K5Top3ReflectedConvolutionComparisonRow("exact Rref", 0.0, None, None, 0.0),
            K5Top3ReflectedConvolutionComparisonRow("old min-type upper", 0.0, None, None, 0.0),
            K5Top3ReflectedConvolutionComparisonRow("new heat-gradient upper", 0.0, None, None, 0.0),
        )
    if source_row is None:
        source_row = _top3_choose_reflected_source_gradient_envelope(
            k5_top3_reflected_source_gaussian_envelope_rows(
                max_n=max_n,
                dense_a_limit=dense_a_limit,
            )
        )
    if source_row is None:
        return ()
    old_rows = {
        row.source_name: row
        for row in k5_top3_reflected_boundary_envelope_rows(
            max_n=max_n,
            dense_a_limit=dense_a_limit,
        )
    }
    old_constant = old_rows["min(1,(a+1)/(t+1))"].source_constant
    source_grid = _top3_reflected_source_grid(max_n=max_n - 1, dense_a_limit=dense_a_limit)
    stats = {
        "exact Rref": {"max_scaled": 0.0, "argmax_n": None, "argmax_a": None, "argmax_value": 0.0},
        "old min-type upper": {"max_scaled": 0.0, "argmax_n": None, "argmax_a": None, "argmax_value": 0.0},
        "new heat-gradient upper": {"max_scaled": 0.0, "argmax_n": None, "argmax_a": None, "argmax_value": 0.0},
    }
    for a in range(1, dense_a_limit + 1):
        kernel = _impl._top3_diagonal_dirichlet_heat_kernel(a, max_n - 1)
        exact_source = np.maximum(source_grid[:max_n, a], 0.0)
        old_source = old_constant * _top3_reflected_convolution_source_profile(a=a, max_n=max_n, source_name="min(1,(a+1)/(t+1))")
        new_source = source_row.source_constant * _top3_reflected_source_gaussian_profile(
            a=a,
            max_n=max_n,
            family_name=source_row.family_name,
            heat_c=source_row.heat_c,
        )
        start_n = max(1, a - 2)
        for n_value in range(start_n, max_n + 1):
            kernel_slice = kernel[:n_value, 0, a - 1][::-1]
            exact_value = 0.5 * float(kernel_slice @ exact_source[:n_value])
            old_upper = 0.5 * float(kernel_slice @ old_source[:n_value])
            new_upper = 0.5 * float(kernel_slice @ new_source[:n_value])
            for name, value in (
                ("exact Rref", exact_value),
                ("old min-type upper", old_upper),
                ("new heat-gradient upper", new_upper),
            ):
                scaled = (n_value + 1) * max(value, 0.0)
                if scaled > stats[name]["max_scaled"]:
                    stats[name]["max_scaled"] = scaled
                    stats[name]["argmax_n"] = n_value
                    stats[name]["argmax_a"] = a
                    stats[name]["argmax_value"] = value
    return tuple(
        K5Top3ReflectedConvolutionComparisonRow(
            name=name,
            max_scaled=float(stat["max_scaled"]),
            argmax_n=stat["argmax_n"],
            argmax_a=stat["argmax_a"],
            argmax_value=float(stat["argmax_value"]),
        )
        for name, stat in stats.items()
    )


def k5_top3_reflected_gradient_convolution_regime_rows(
    max_n: int = 5000,
    dense_a_limit: int = 1000,
    kernel_heat_c: float = 2.0,
    kernel_constant: float = 1.0,
    source_heat_c: float = 2.0,
    source_constant: float = 1.0,
) -> tuple[K5Top3ReflectedGradientConvolutionRegimeRow, ...]:
    labels = ("short-s / early-t", "short-s / late-t", "long-s / early-t", "long-s / late-t")
    stats = {
        label: {"max_scaled_upper": 0.0, "argmax_n": None, "argmax_a": None}
        for label in labels
    }
    n_weights = np.arange(max_n + 1, dtype=float) + 1.0
    for a in range(1, dense_a_limit + 1):
        heat_profile = _top3_reflected_convolution_heat_profile(a=a, max_n=max_n, heat_c=kernel_heat_c)
        source_profile = _top3_reflected_source_gaussian_profile(
            a=a,
            max_n=max_n,
            family_name="(a+1)/(t+1)^(3/2)*exp",
            heat_c=source_heat_c,
        )
        threshold = min(max_n, a * a)

        short_s = np.zeros_like(heat_profile)
        short_s[1 : threshold + 1] = heat_profile[1 : threshold + 1]
        long_s = heat_profile - short_s

        early_t = np.zeros_like(source_profile)
        early_t[:threshold] = source_profile[:threshold]
        late_t = source_profile - early_t

        regime_profiles = {
            "short-s / early-t": _top3_reflected_convolution_fft(short_s, early_t, max_n + 1),
            "short-s / late-t": _top3_reflected_convolution_fft(short_s, late_t, max_n + 1),
            "long-s / early-t": _top3_reflected_convolution_fft(long_s, early_t, max_n + 1),
            "long-s / late-t": _top3_reflected_convolution_fft(long_s, late_t, max_n + 1),
        }
        start_n = max(0, a - 2)
        for label, profile in regime_profiles.items():
            scaled = 0.5 * kernel_constant * source_constant * n_weights * np.maximum(profile, 0.0)
            local_index = int(np.argmax(scaled[start_n:])) if start_n <= max_n else 0
            n_value = start_n + local_index
            value = float(scaled[n_value])
            if value > stats[label]["max_scaled_upper"]:
                stats[label]["max_scaled_upper"] = value
                stats[label]["argmax_n"] = n_value
                stats[label]["argmax_a"] = a
    return tuple(
        K5Top3ReflectedGradientConvolutionRegimeRow(
            label=label,
            max_scaled_upper=float(stats[label]["max_scaled_upper"]),
            argmax_n=stats[label]["argmax_n"],
            argmax_a=stats[label]["argmax_a"],
        )
        for label in labels
    )


def k5_top3_reflected_convolution_analytic_scan_rows(
    max_n: int = 500,
    dense_a_limit: int = 200,
    verification_max_n: int = 120,
    verification_a_limit: int = 60,
    heat_c_values: tuple[float, ...] = (4.0, 8.0, 16.0),
    source_name: str = "min(1,(a+1)/(t+1))",
) -> tuple[K5Top3ReflectedConvolutionAnalyticScanRow, ...]:
    kernel_rows = k5_top3_reflected_boundary_kernel_estimate_rows(
        max_n=verification_max_n,
        dense_a_limit=verification_a_limit,
        heat_c_values=heat_c_values,
    )
    source_rows = {
        row.source_name: row
        for row in k5_top3_reflected_boundary_envelope_rows(
            max_n=verification_max_n,
            dense_a_limit=verification_a_limit,
        )
    }
    if source_name not in source_rows:
        raise ValueError(f"unknown source envelope {source_name!r}")
    source_constant = source_rows[source_name].source_constant
    n_weights = np.arange(max_n + 1, dtype=float) + 1.0
    rows: list[K5Top3ReflectedConvolutionAnalyticScanRow] = []
    for kernel_row in kernel_rows:
        unit_max_scaled = 0.0
        argmax_n: int | None = None
        argmax_a: int | None = None
        for a in range(1, dense_a_limit + 1):
            heat_profile = _top3_reflected_convolution_heat_profile(a=a, max_n=max_n, heat_c=kernel_row.heat_c)
            source_profile = _top3_reflected_convolution_source_profile(a=a, max_n=max_n, source_name=source_name)
            convolved = _top3_reflected_convolution_fft(heat_profile, source_profile, max_n + 1)
            scaled = 0.5 * n_weights * np.maximum(convolved, 0.0)
            start_n = max(0, a - 2)
            local_index = int(np.argmax(scaled[start_n:])) if start_n <= max_n else 0
            n_value = start_n + local_index
            value = float(scaled[n_value])
            if value > unit_max_scaled:
                unit_max_scaled = value
                argmax_n = n_value
                argmax_a = a
        rows.append(
            K5Top3ReflectedConvolutionAnalyticScanRow(
                heat_c=kernel_row.heat_c,
                kernel_constant=kernel_row.kernel_constant,
                source_constant=source_constant,
                unit_max_scaled=unit_max_scaled,
                max_scaled_upper=kernel_row.kernel_constant * source_constant * unit_max_scaled,
                argmax_n=argmax_n,
                argmax_a=argmax_a,
            )
        )
    return tuple(rows)


def k5_top3_reflected_convolution_regime_rows(
    max_n: int = 500,
    dense_a_limit: int = 200,
    heat_c: float = 8.0,
    kernel_constant: float = 1.0,
    source_constant: float = 1.0,
) -> tuple[K5Top3ReflectedConvolutionRegimeRow, ...]:
    labels = ("early-short", "early-long", "late-short", "late-long")
    stats = {
        label: {
            "max_scaled_upper": 0.0,
            "argmax_n": None,
            "argmax_a": None,
        }
        for label in labels
    }
    n_weights = np.arange(max_n + 1, dtype=float) + 1.0
    for a in range(1, dense_a_limit + 1):
        heat_profile = _top3_reflected_convolution_heat_profile(a=a, max_n=max_n, heat_c=heat_c)
        source_profile = _top3_reflected_convolution_source_profile(a=a, max_n=max_n)
        short_cutoff = min(max_n, a * a)

        heat_short = np.zeros_like(heat_profile)
        heat_short[1 : short_cutoff + 1] = heat_profile[1 : short_cutoff + 1]
        heat_long = heat_profile - heat_short

        source_early = np.zeros_like(source_profile)
        early_cutoff = min(max_n, a + 1)
        source_early[:early_cutoff] = 1.0
        source_late = source_profile - source_early

        regime_profiles = {
            "early-short": _top3_reflected_convolution_fft(heat_short, source_early, max_n + 1),
            "early-long": _top3_reflected_convolution_fft(heat_long, source_early, max_n + 1),
            "late-short": _top3_reflected_convolution_fft(heat_short, source_late, max_n + 1),
            "late-long": _top3_reflected_convolution_fft(heat_long, source_late, max_n + 1),
        }
        start_n = max(0, a - 2)
        for label, profile in regime_profiles.items():
            scaled = 0.5 * kernel_constant * source_constant * n_weights * np.maximum(profile, 0.0)
            local_index = int(np.argmax(scaled[start_n:])) if start_n <= max_n else 0
            n_value = start_n + local_index
            value = float(scaled[n_value])
            if value > stats[label]["max_scaled_upper"]:
                stats[label]["max_scaled_upper"] = value
                stats[label]["argmax_n"] = n_value
                stats[label]["argmax_a"] = a
    return tuple(
        K5Top3ReflectedConvolutionRegimeRow(
            label=label,
            max_scaled_upper=float(stats[label]["max_scaled_upper"]),
            argmax_n=stats[label]["argmax_n"],
            argmax_a=stats[label]["argmax_a"],
        )
        for label in labels
    )


def print_k5_top3_reflected_boundary_convolution_report(
    max_n: int = 500,
    dense_a_limit: int = 200,
) -> None:
    verification_max_n = min(max_n, 120)
    verification_a_limit = min(dense_a_limit, 60)
    print("reflected-boundary convolution report")
    print()
    print(f"verification grid: n <= {verification_max_n}, 1 <= a <= min(n+2,{verification_a_limit})")
    print(f"scan grid: n <= {max_n}, 1 <= a <= min(n+2,{dense_a_limit})")
    print("No packet LP and no broad cone search are used here; this is a symbolic reflected-boundary proof-helper report.")

    print()
    print("1. exact Rref formula")
    print(r"Define the reflected-boundary contribution")
    print(r"Rref(n,a)=\frac12\sum_{t=0}^{n-1}K_{n-1-t}^{(a)}(1,a)E_{a+1}^{\mathrm{ref}}(t,a),")
    print(r"where E_{a+1}^{\mathrm{ref}}(t,a)=(-1)^{t+1}\bigl(p_t(1,a+1)-p_t(0,a+1)\bigr).")
    print(r"Then E_1(n,a)=\text{initial}(n,a)+\text{root}(n,a)+Rref(n,a).")

    print()
    print("2. Duhamel decomposition verification")
    print("a max_abs_residual argmax_n residual_at_arg")
    residual_rows = k5_top3_reflected_boundary_convolution_residual_rows(
        max_n=verification_max_n,
        dense_a_limit=verification_a_limit,
    )
    for row in residual_rows[: min(len(residual_rows), 12)]:
        print(
            f"{row.a:2d}"
            f" {row.max_abs_residual:.3e}"
            f" {row.arg_n}"
            f" {row.residual_at_arg:+.3e}"
        )
    worst_row = max(residual_rows, key=lambda item: item.max_abs_residual) if residual_rows else None
    if worst_row is not None:
        print(
            f"worst decomposition residual: {worst_row.max_abs_residual:.3e}"
            f" at a={worst_row.a}, n={worst_row.arg_n}"
        )

    print()
    print("3. dense contribution scan")
    print("name min_value max_(n+1)*value argmax sign_holds")
    scan_rows = k5_top3_reflected_boundary_convolution_scan_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
    )
    scan_by_name = {row.name: row for row in scan_rows}
    for name in ("Rref", "root", "initial", "E1"):
        row = scan_by_name[name]
        print(
            f"{row.name:7s}"
            f" {row.min_value:.12f}"
            f" {row.max_scaled:.12f}"
            f" ({row.argmax_n},{row.argmax_a})"
            f" {row.sign_holds}"
        )
    rref_row = scan_by_name["Rref"]
    root_row = scan_by_name["root"]
    root_ratio = rref_row.max_scaled / root_row.max_scaled if root_row.max_scaled > 0.0 else float("inf")
    print(
        f"max normalized reflected contribution: {rref_row.max_scaled:.12f}"
        f" at (n,a)=({rref_row.argmax_n},{rref_row.argmax_a})"
    )
    print(
        f"comparison to root contribution: max_Rref / max_root = {root_ratio:.12f}"
        f" with max_root = {root_row.max_scaled:.12f}"
    )

    print()
    print("4. a/sqrt(n) bins")
    print("bin count max_(n+1)Rref argmax ratio sign_holds")
    bin_rows = k5_top3_reflected_boundary_convolution_bin_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
    )
    for row in bin_rows:
        print(
            f"{row.label:8s}"
            f" {row.count:6d}"
            f" {row.max_scaled:.12f}"
            f" ({row.argmax_n},{row.argmax_a})"
            f" {row.argmax_ratio:.6f}"
            f" {row.sign_holds}"
        )

    print()
    print("5. candidate heat-kernel envelope")
    print(r"Test K_m^{(a)}(1,a) \le C_{heat}\, a\,(m+1)^{-3/2}\exp\!\bigl(-a^2/(c\,(m+1))\bigr).")
    print("c kernel_constant")
    kernel_rows = k5_top3_reflected_boundary_kernel_estimate_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
    )
    for row in kernel_rows:
        print(f"{row.heat_c:4.1f} {row.kernel_constant:.12f}")
    pointwise_row = k5_top3_reflected_boundary_pointwise_row(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
    )
    print(
        f"max_(n+1)^2 single summand = {pointwise_row.max_n2_summand:.12f}"
        f" at (n,a,t)=({pointwise_row.argmax_n},{pointwise_row.argmax_a},{pointwise_row.argmax_t})"
    )
    print(
        f"single summand value there = {pointwise_row.argmax_value:.12e};"
        " the pointwise kernel-source product is consistent with an O((n+1)^-2) summable variant on the scanned grid."
    )

    print()
    print("6. theorem-ready source envelopes")
    print("candidate envelope C_ref max_(n+1) upper argmax")
    envelope_rows = k5_top3_reflected_boundary_envelope_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
    )
    for row in envelope_rows:
        print(
            f"{row.source_name:24s}"
            f" {row.source_constant:.12f}"
            f" {row.max_scaled_upper:.12f}"
            f" ({row.argmax_n},{row.argmax_a})"
        )
    best_envelope = min(envelope_rows, key=lambda item: item.max_scaled_upper) if envelope_rows else None
    if best_envelope is not None:
        print(
            "smallest scanned sufficient source envelope:"
            f" {best_envelope.source_name}"
            f" with C_ref={best_envelope.source_constant:.12f}"
            f" and max_(n+1) upper={best_envelope.max_scaled_upper:.12f}"
        )

    print()
    print("7. implication for E_1")
    print(r"Since E_1(n,a)=\text{initial}(n,a)+\text{root}(n,a)+Rref(n,a), any uniform O((n+1)^{-1}) bound for Rref combines with the already isolated root convolution bound to give E_1(n,a)=O((n+1)^{-1}).")
    print(r"A candidate analytic inequality is")
    if kernel_rows and best_envelope is not None:
        chosen_kernel = min(kernel_rows, key=lambda item: item.kernel_constant)
        print(
            r"K_m^{(a)}(1,a) \le "
            f"{chosen_kernel.kernel_constant:.6f}"
            r"\, a\,(m+1)^{-3/2}\exp\!\bigl(-a^2/("
            f"{chosen_kernel.heat_c:.1f}"
            r"\,(m+1))\bigr),"
        )
        print(
            r"E_{a+1}^{\mathrm{ref}}(t,a) \le "
            f"{best_envelope.source_constant:.6f}"
            r"\,"
            f"{best_envelope.source_name},"
        )
        print(r"which numerically yields (n+1)Rref(n,a) uniformly bounded on the full scan.")
    else:
        print(r"K_m^{(a)}(1,a) with a Gaussian heat-kernel tail, together with a uniform O((a+1)/(t+1)) or min-type bound for E_{a+1}^{\mathrm{ref}}(t,a), suffices to control Rref.")

    print()
    print("8. theorem-ready statements")
    print("reflected-boundary convolution")
    print(r"Rref(n,a)=\frac12\sum_{t=0}^{n-1}K_{n-1-t}^{(a)}(1,a)E_{a+1}^{\mathrm{ref}}(t,a).")
    print("candidate envelope")
    print(r"It is enough to prove E_{a+1}^{\mathrm{ref}}(t,a) \le C_{ref}\min\!\bigl(1,(a+1)/(t+1)\bigr), together with the Gaussian finite-interval kernel bound for K_m^{(a)}(1,a).")
    print("Rref")
    print(r"Under those bounds, the reflected-boundary convolution is consistent with Rref(n,a)=O((n+1)^{-1}).")
    print("implication for E_1")
    print(r"Substituting into E_1(n,a)=\text{initial}+\text{root}+Rref gives E_1(n,a)=O((n+1)^{-1}), hence the same O((n+1)^{-1}) route for the exact K image away from a=1.")


def print_k5_top3_reflected_convolution_analytic_bound_report(
    max_n: int = 500,
    dense_a_limit: int = 200,
) -> None:
    verification_max_n = min(max_n, 120)
    verification_a_limit = min(dense_a_limit, 60)
    print("reflected convolution analytic bound report")
    print()
    print(f"verification grid: n <= {verification_max_n}, 1 <= a <= min(n+2,{verification_a_limit})")
    print(f"abstract scan grid: n <= {max_n}, 1 <= a <= min(n+2,{dense_a_limit})")
    print("The exact Dirichlet-kernel checks stay on the capped verification grid; the full scan uses only abstract envelope convolutions.")
    print("No packet LP and no broad cone search are used here.")

    print()
    print("1. exact reflected-boundary calibration")
    print("name min_value max_(n+1)*value argmax sign_holds")
    exact_rows = {
        row.name: row
        for row in k5_top3_reflected_boundary_convolution_scan_rows(
            max_n=verification_max_n,
            dense_a_limit=verification_a_limit,
        )
    }
    for name in ("Rref", "root", "initial", "E1"):
        row = exact_rows[name]
        print(
            f"{row.name:7s}"
            f" {row.min_value:.12f}"
            f" {row.max_scaled:.12f}"
            f" ({row.argmax_n},{row.argmax_a})"
            f" {row.sign_holds}"
        )

    print()
    print("2. kernel bound")
    print(r"Calibrate K_m^{(a)}(1,a) \le C_{heat}\, a\,(m+1)^{-3/2}\exp\!\bigl(-a^2/(c\,(m+1))\bigr) on the verification grid.")
    print("c C_heat")
    kernel_rows = k5_top3_reflected_boundary_kernel_estimate_rows(
        max_n=verification_max_n,
        dense_a_limit=verification_a_limit,
        heat_c_values=(4.0, 8.0, 16.0),
    )
    for row in kernel_rows:
        print(f"{row.heat_c:4.1f} {row.kernel_constant:.12f}")

    print()
    print("3. source envelope")
    print(r"Calibrate E_{a+1}^{\mathrm{ref}}(t,a) \le C_{ref}\min\!\bigl(1,(a+1)/(t+1)\bigr) and compare to the weaker pure harmonic envelope.")
    print("candidate C_ref max_(n+1) exact-upper argmax")
    envelope_rows = k5_top3_reflected_boundary_envelope_rows(
        max_n=verification_max_n,
        dense_a_limit=verification_a_limit,
    )
    envelope_by_name = {row.source_name: row for row in envelope_rows}
    for row in envelope_rows:
        print(
            f"{row.source_name:24s}"
            f" {row.source_constant:.12f}"
            f" {row.max_scaled_upper:.12f}"
            f" ({row.argmax_n},{row.argmax_a})"
        )

    print()
    print("4. abstract convolution inequality")
    print(r"Scan S(n,a)=\frac12\sum_{t=0}^{n-1} a\,(n-t)^{-3/2}\exp\!\bigl(-a^2/(c\,(n-t))\bigr)\min\!\bigl(1,(a+1)/(t+1)\bigr) via FFT convolution, then multiply by the scanned constants C_{heat}C_{ref}.")
    print("c unit_max_(n+1)S max_(n+1) upper argmax")
    analytic_rows = k5_top3_reflected_convolution_analytic_scan_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
        verification_max_n=verification_max_n,
        verification_a_limit=verification_a_limit,
        heat_c_values=(4.0, 8.0, 16.0),
    )
    for row in analytic_rows:
        print(
            f"{row.heat_c:4.1f}"
            f" {row.unit_max_scaled:.12f}"
            f" {row.max_scaled_upper:.12f}"
            f" ({row.argmax_n},{row.argmax_a})"
        )
    chosen_row = min(analytic_rows, key=lambda item: item.max_scaled_upper) if analytic_rows else None

    print()
    print("5. regime split")
    print(r"Split the abstract convolution into source-early/source-late and heat-short/heat-long pieces, where early means t\le a and short means n-t\le a^2.")
    print("regime max_(n+1) upper argmax")
    if chosen_row is not None:
        regime_rows = k5_top3_reflected_convolution_regime_rows(
            max_n=max_n,
            dense_a_limit=dense_a_limit,
            heat_c=chosen_row.heat_c,
            kernel_constant=chosen_row.kernel_constant,
            source_constant=chosen_row.source_constant,
        )
        for row in regime_rows:
            print(
                f"{row.label:11s}"
                f" {row.max_scaled_upper:.12f}"
                f" ({row.argmax_n},{row.argmax_a})"
            )

    print()
    print("6. theorem-ready corollaries")
    if chosen_row is not None:
        source_row = envelope_by_name["min(1,(a+1)/(t+1))"]
        print("chosen kernel/source pair")
        print(
            r"K_m^{(a)}(1,a) \le "
            f"{chosen_row.kernel_constant:.6f}"
            r"\, a\,(m+1)^{-3/2}\exp\!\bigl(-a^2/("
            f"{chosen_row.heat_c:.1f}"
            r"\,(m+1))\bigr),"
        )
        print(
            r"E_{a+1}^{\mathrm{ref}}(t,a) \le "
            f"{source_row.source_constant:.6f}"
            r"\min\!\bigl(1,(a+1)/(t+1)\bigr)."
        )
        print(
            r"Consequently the scanned convolution inequality gives (n+1)Rref(n,a) \le "
            f"{chosen_row.max_scaled_upper:.6f}"
            r" on the abstract scan grid."
        )
        print(r"Combining this with the already-isolated initial and root terms supports E_1(n,a)=O((n+1)^{-1}), hence the same O((n+1)^{-1}) route for the exact K image away from a=1.")
    else:
        print(r"A Gaussian kernel envelope together with E_{a+1}^{\mathrm{ref}}(t,a) \le C_{ref}\min\!\bigl(1,(a+1)/(t+1)\bigr) is the desired analytic route for Rref(n,a)=O((n+1)^{-1}).")


def print_k5_top3_reflected_source_gaussian_report(
    max_n: int = 5000,
    dense_a_limit: int = 1000,
) -> None:
    verification_max_n = min(max_n, 120)
    verification_a_limit = min(dense_a_limit, 60)
    source_grid = _top3_reflected_source_grid(max_n=max_n, dense_a_limit=dense_a_limit)
    summary_row = k5_top3_reflected_source_summary_row(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
        source_grid=source_grid,
    )
    bin_rows = k5_top3_reflected_source_bin_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
        source_grid=source_grid,
    )
    envelope_rows = k5_top3_reflected_source_gaussian_envelope_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
        source_grid=source_grid,
    )
    best_source_row = _top3_choose_reflected_source_gaussian_envelope(envelope_rows)
    new_convolution_rows = k5_top3_reflected_convolution_gaussian_scan_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
        source_row=best_source_row,
        verification_max_n=verification_max_n,
        verification_a_limit=verification_a_limit,
    )
    exact_comparison_rows = k5_top3_reflected_convolution_exact_upper_comparison_rows(
        max_n=verification_max_n,
        dense_a_limit=verification_a_limit,
        source_row=best_source_row,
    )
    print("reflected source gaussian report")
    print()
    print(f"source scan grid: t <= {max_n}, 1 <= a <= min(t+2,{dense_a_limit})")
    print(f"exact kernel verification grid: n <= {verification_max_n}, 1 <= a <= min(n+2,{verification_a_limit})")
    print("No packet LP and no broad cone search are used here; this is a symbolic/proof-helper reflected source report.")

    print()
    print("1. exact reflected source scan")
    print(r"E_ref(t,a)=(-1)^{t+1}\bigl(p_t(1,a+1)-p_t(0,a+1)\bigr).")
    print(
        f"max E_ref = {summary_row.max_value:.12f}"
        f" at (t,a)=({summary_row.argmax_value_t},{summary_row.argmax_value_a})"
    )
    print(
        f"max (t+1)E_ref = {summary_row.max_t_scaled:.12f}"
        f" at (t,a)=({summary_row.argmax_t_scaled_t},{summary_row.argmax_t_scaled_a})"
    )
    print(
        f"max (t+1)E_ref/(a+1) = {summary_row.max_t_over_a_scaled:.12f}"
        f" at (t,a)=({summary_row.argmax_t_over_a_scaled_t},{summary_row.argmax_t_over_a_scaled_a})"
    )

    print()
    print("2. bins by a/sqrt(t+1)")
    print("bin count max_E_ref max_(t+1)E_ref argmax ratio")
    for row in bin_rows:
        print(
            f"{row.label:8s}"
            f" {row.count:7d}"
            f" {row.max_value:.12f}"
            f" {row.max_t_scaled:.12f}"
            f" ({row.argmax_t},{row.argmax_a})"
            f" {row.argmax_ratio:.6f}"
        )

    print()
    print("3. Gaussian source envelope fits")
    print("family c C_source argmax ratio hit_t_boundary hit_a_boundary")
    for row in envelope_rows:
        print(
            f"{row.family_name:30s}"
            f" {row.heat_c:4.1f}"
            f" {row.source_constant:.12f}"
            f" ({row.argmax_t},{row.argmax_a})"
            f" {row.argmax_ratio:.6f}"
            f" {row.hits_t_boundary}"
            f" {row.hits_a_boundary}"
        )

    print()
    print("4. old envelope too crude")
    old_rows = {
        row.source_name: row
        for row in k5_top3_reflected_boundary_envelope_rows(
            max_n=verification_max_n,
            dense_a_limit=verification_a_limit,
        )
    }
    old_row = old_rows["min(1,(a+1)/(t+1))"]
    print(
        "old envelope:"
        f" C_ref={old_row.source_constant:.12f}"
        f" and max_(n+1) upper={old_row.max_scaled_upper:.12f}"
        f" on the exact kernel verification grid"
    )
    print("The old envelope too crude signal is that its convolution upper peaks near the scan edge and stays much larger than the exact reflected term.")

    print()
    print("5. tightest plausible theorem envelope")
    if best_source_row is not None:
        print(
            f"chosen family: {best_source_row.family_name}"
            f" with c={best_source_row.heat_c:.1f}"
            f" and C_source={best_source_row.source_constant:.12f}"
        )
        print(
            f"argmax at (t,a)=({best_source_row.argmax_t},{best_source_row.argmax_a})"
            f" with a/sqrt(t+1)={best_source_row.argmax_ratio:.6f}"
            f"; hit_t_boundary={best_source_row.hits_t_boundary}, hit_a_boundary={best_source_row.hits_a_boundary}"
        )
    else:
        print("No finite Gaussian source envelope was found on the scanned grid.")

    print()
    print("6. new convolution upper")
    print("kernel_c unit_max_(n+1)S max_(n+1) upper argmax")
    for row in new_convolution_rows:
        print(
            f"{row.kernel_heat_c:7.1f}"
            f" {row.unit_max_scaled:.12f}"
            f" {row.max_scaled_upper:.12f}"
            f" ({row.argmax_n},{row.argmax_a})"
        )
    best_convolution_row = min(new_convolution_rows, key=lambda item: item.max_scaled_upper) if new_convolution_rows else None
    if best_convolution_row is not None:
        print(
            f"best new convolution upper: max_(n+1)S(n,a) <= {best_convolution_row.max_scaled_upper:.12f}"
            f" using kernel c={best_convolution_row.kernel_heat_c:.1f}"
        )

    print()
    print("7. exact convolution comparison")
    print("name max_(n+1)value argmax value")
    for row in exact_comparison_rows:
        print(
            f"{row.name:18s}"
            f" {row.max_scaled:.12f}"
            f" ({row.argmax_n},{row.argmax_a})"
            f" {row.argmax_value:.12f}"
        )

    print()
    print("8. theorem-ready candidates")
    if best_source_row is not None and best_convolution_row is not None:
        print("Lemma 1: source Gaussian envelope")
        print(
            r"E_{a+1}^{\mathrm{ref}}(t,a) \le "
            f"{best_source_row.source_constant:.6f}"
            r"\,"
            f"{best_source_row.family_name.replace('*exp', '')}"
            r"\exp\!\bigl(-a^2/("
            f"{best_source_row.heat_c:.1f}"
            r"\,(t+1))\bigr)."
        )
        print("Lemma 2: kernel Gaussian envelope")
        print(
            r"K_m^{(a)}(1,a) \le "
            f"{best_convolution_row.kernel_constant:.6f}"
            r"\, a\,(m+1)^{-3/2}\exp\!\bigl(-a^2/("
            f"{best_convolution_row.kernel_heat_c:.1f}"
            r"\,(m+1))\bigr)."
        )
        print("Lemma 3: product-convolution is O(1/n)")
        print(
            r"Combining the two Gaussian envelopes gives (n+1)S(n,a) \le "
            f"{best_convolution_row.max_scaled_upper:.6f}"
            r" on the scanned grid, consistent with S(n,a)=O((n+1)^{-1})."
        )
    else:
        print("A finite Gaussian source envelope plus the existing kernel Gaussian envelope remains the candidate route for the reflected convolution.")


def print_k5_top3_reflected_gradient_convolution_report(
    max_n: int = 5000,
    dense_a_limit: int = 1000,
) -> None:
    verification_max_n = min(max_n, 120)
    verification_a_limit = min(dense_a_limit, 60)
    source_grid = _top3_reflected_source_grid(max_n=max_n, dense_a_limit=dense_a_limit)
    source_rows = k5_top3_reflected_source_gaussian_envelope_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
        source_grid=source_grid,
    )
    gradient_rows = tuple(
        row for row in source_rows if row.family_name == "(a+1)/(t+1)^(3/2)*exp"
    )
    best_source_row = _top3_choose_reflected_source_gradient_envelope(gradient_rows)
    kernel_rows = k5_top3_reflected_boundary_kernel_estimate_rows(
        max_n=verification_max_n,
        dense_a_limit=verification_a_limit,
        heat_c_values=(2.0, 4.0, 8.0, 16.0, 32.0),
    )
    convolution_rows = k5_top3_reflected_gradient_convolution_scan_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
        source_rows=source_rows,
        verification_max_n=verification_max_n,
        verification_a_limit=verification_a_limit,
    )
    exact_rows = {
        row.name: row
        for row in k5_top3_reflected_boundary_convolution_scan_rows(
            max_n=verification_max_n,
            dense_a_limit=verification_a_limit,
        )
    }
    old_analytic_rows = k5_top3_reflected_convolution_analytic_scan_rows(
        max_n=max_n,
        dense_a_limit=dense_a_limit,
        verification_max_n=verification_max_n,
        verification_a_limit=verification_a_limit,
        heat_c_values=(2.0, 4.0, 8.0, 16.0, 32.0),
        source_name="min(1,(a+1)/(t+1))",
    )
    print("reflected gradient convolution report")
    print()
    print(f"source/convolution scan grid: n <= {max_n}, 1 <= a <= min(n+2,{dense_a_limit})")
    print(f"exact kernel verification grid: n <= {verification_max_n}, 1 <= a <= min(n+2,{verification_a_limit})")
    print("No packet LP and no broad cone search are used here; this is a focused reflected-gradient proof-helper report.")

    print()
    print("1. source gradient envelope")
    print(r"Test E_ref(t,a) \le C_src\,(a+1)(t+1)^{-3/2}\exp\!\bigl(-a^2/(c_src(t+1))\bigr).")
    print("c_src C_src argmax ratio hit_t_boundary hit_a_boundary")
    for row in gradient_rows:
        print(
            f"{row.heat_c:5.1f}"
            f" {row.source_constant:.12f}"
            f" ({row.argmax_t},{row.argmax_a})"
            f" {row.argmax_ratio:.6f}"
            f" {row.hits_t_boundary}"
            f" {row.hits_a_boundary}"
        )

    print()
    print("2. kernel Gaussian envelope")
    print(r"Test K_m^{(a)}(1,a) \le C_heat\, a\,(m+1)^{-3/2}\exp\!\bigl(-a^2/(c_heat(m+1))\bigr).")
    print("c_heat C_heat")
    for row in kernel_rows:
        print(f"{row.heat_c:6.1f} {row.kernel_constant:.12f}")

    print()
    print("3. gradient convolution upper")
    print("c_heat c_src unit_max_(n+1)S max_(n+1)S argmax")
    for row in convolution_rows:
        print(
            f"{row.kernel_heat_c:6.1f}"
            f" {row.source_heat_c:5.1f}"
            f" {row.unit_max_scaled:.12f}"
            f" {row.max_scaled_upper:.12f}"
            f" ({row.argmax_n},{row.argmax_a})"
        )
    gradient_convolution_rows = tuple(
        row for row in convolution_rows if row.source_family_name != "exact-reflected-source"
    )
    best_convolution_row = (
        min(gradient_convolution_rows, key=lambda item: item.max_scaled_upper)
        if gradient_convolution_rows
        else None
    )
    best_old_analytic_row = min(old_analytic_rows, key=lambda item: item.max_scaled_upper) if old_analytic_rows else None

    print()
    print("4. comparison of uppers")
    print("quantity max_(n+1)value argmax")
    exact_rref_row = exact_rows["Rref"]
    print(
        f"{'exact Rref':23s}"
        f" {exact_rref_row.max_scaled:.12f}"
        f" ({exact_rref_row.argmax_n},{exact_rref_row.argmax_a})"
    )
    if best_old_analytic_row is not None:
        print(
            f"{'old min-type upper':23s}"
            f" {best_old_analytic_row.max_scaled_upper:.12f}"
            f" ({best_old_analytic_row.argmax_n},{best_old_analytic_row.argmax_a})"
        )
    if best_convolution_row is not None:
        print(
            f"{'new heat-gradient upper':23s}"
            f" {best_convolution_row.max_scaled_upper:.12f}"
            f" ({best_convolution_row.argmax_n},{best_convolution_row.argmax_a})"
        )

    print()
    print("5. proof split")
    if best_convolution_row is not None and best_source_row is not None:
        regime_rows = k5_top3_reflected_gradient_convolution_regime_rows(
            max_n=max_n,
            dense_a_limit=dense_a_limit,
            kernel_heat_c=best_convolution_row.kernel_heat_c,
            kernel_constant=best_convolution_row.kernel_constant,
            source_heat_c=best_source_row.heat_c,
            source_constant=best_source_row.source_constant,
        )
        print("regime max_(n+1) upper argmax")
        for row in regime_rows:
            print(
                f"{row.label:18s}"
                f" {row.max_scaled_upper:.12f}"
                f" ({row.argmax_n},{row.argmax_a})"
            )
        dominant_row = max(regime_rows, key=lambda item: item.max_scaled_upper) if regime_rows else None
        if dominant_row is not None:
            print(f"dominant regime: {dominant_row.label}")

    print()
    print("6. analytic object")
    print(r"Numerically test")
    print(r"a(a+1)\sum_{s=1}^{n} s^{-3/2}(n-s+1)^{-3/2}\exp\!\bigl(-a^2/(c_1 s)-a^2/(c_2(n-s+1))\bigr) \le C/(n+1).")
    if best_convolution_row is not None and best_source_row is not None:
        print(
            f"best scanned pair: c_heat={best_convolution_row.kernel_heat_c:.1f}, c_src={best_source_row.heat_c:.1f},"
            f" max_(n+1)S={best_convolution_row.max_scaled_upper:.12f}"
            f" at (n,a)=({best_convolution_row.argmax_n},{best_convolution_row.argmax_a})"
        )
        if best_old_analytic_row is not None:
            print(
                f"best old abstract min-type upper: {best_old_analytic_row.max_scaled_upper:.12f}"
                f" at (n,a)=({best_old_analytic_row.argmax_n},{best_old_analytic_row.argmax_a})"
            )

    print()
    print("7. candidate lemmas")
    if best_convolution_row is not None and best_source_row is not None:
        print("best constants")
        print(
            f"C_src={best_source_row.source_constant:.12f}, c_src={best_source_row.heat_c:.1f};"
            f" C_heat={best_convolution_row.kernel_constant:.12f}, c_heat={best_convolution_row.kernel_heat_c:.1f}."
        )
        print(
            f"source max hits boundary: t={best_source_row.hits_t_boundary}, a={best_source_row.hits_a_boundary};"
            f" convolution max at (n,a)=({best_convolution_row.argmax_n},{best_convolution_row.argmax_a})."
        )
        print("Lemma 1: source gradient envelope")
        print(
            r"E_{a+1}^{\mathrm{ref}}(t,a) \le "
            f"{best_source_row.source_constant:.6f}"
            r"\,(a+1)(t+1)^{-3/2}\exp\!\bigl(-a^2/("
            f"{best_source_row.heat_c:.1f}"
            r"\,(t+1))\bigr)."
        )
        print("Lemma 2: kernel Gaussian envelope")
        print(
            r"K_m^{(a)}(1,a) \le "
            f"{best_convolution_row.kernel_constant:.6f}"
            r"\, a(m+1)^{-3/2}\exp\!\bigl(-a^2/("
            f"{best_convolution_row.kernel_heat_c:.1f}"
            r"\,(m+1))\bigr)."
        )
        print("Lemma 3: product-convolution is O(1/n)")
        print(
            r"Combining these gives S(n,a) \le C/(n+1), with scanned constant C="
            f"{best_convolution_row.max_scaled_upper:.6f}."
        )
    else:
        print("The heat-gradient source family is the candidate proof route once paired with the finite-interval kernel Gaussian envelope.")


__all__ = [
    "K5Top3ReflectedBoundaryConvolutionResidualRow",
    "K5Top3ReflectedBoundaryConvolutionScanRow",
    "K5Top3ReflectedBoundaryConvolutionBinRow",
    "K5Top3ReflectedBoundaryKernelEstimateRow",
    "K5Top3ReflectedBoundaryEnvelopeRow",
    "K5Top3ReflectedBoundaryPointwiseRow",
    "K5Top3ReflectedConvolutionAnalyticScanRow",
    "K5Top3ReflectedConvolutionRegimeRow",
    "K5Top3ReflectedSourceSummaryRow",
    "K5Top3ReflectedSourceBinRow",
    "K5Top3ReflectedSourceGaussianEnvelopeRow",
    "K5Top3ReflectedConvolutionGaussianScanRow",
    "K5Top3ReflectedConvolutionComparisonRow",
    "K5Top3ReflectedGradientConvolutionRegimeRow",
    "k5_top3_reflected_boundary_convolution_residual_rows",
    "k5_top3_reflected_boundary_convolution_scan_rows",
    "k5_top3_reflected_boundary_convolution_bin_rows",
    "k5_top3_reflected_boundary_kernel_estimate_rows",
    "k5_top3_reflected_boundary_pointwise_row",
    "k5_top3_reflected_boundary_envelope_rows",
    "_top3_reflected_boundary_envelope_specs",
    "_top3_reflected_source_bin_specs",
    "_top3_reflected_source_gaussian_family_specs",
    "_top3_reflected_source_gaussian_envelope_value",
    "_top3_reflected_source_gaussian_profile",
    "_top3_reflected_source_grid",
    "k5_top3_reflected_source_summary_row",
    "k5_top3_reflected_source_bin_rows",
    "k5_top3_reflected_source_gaussian_envelope_rows",
    "_top3_choose_reflected_source_gaussian_envelope",
    "_top3_choose_reflected_source_gradient_envelope",
    "_top3_reflected_convolution_fft",
    "_top3_reflected_convolution_heat_profile",
    "_top3_reflected_convolution_source_profile",
    "k5_top3_reflected_convolution_gaussian_scan_rows",
    "k5_top3_reflected_gradient_convolution_scan_rows",
    "k5_top3_reflected_convolution_exact_upper_comparison_rows",
    "k5_top3_reflected_gradient_convolution_comparison_rows",
    "k5_top3_reflected_gradient_convolution_regime_rows",
    "k5_top3_reflected_convolution_analytic_scan_rows",
    "k5_top3_reflected_convolution_regime_rows",
    "print_k5_top3_reflected_boundary_convolution_report",
    "print_k5_top3_reflected_convolution_analytic_bound_report",
    "print_k5_top3_reflected_source_gaussian_report",
    "print_k5_top3_reflected_gradient_convolution_report",
]
