#!/usr/bin/env python
"""
Standalone equivalence checker for the QICK circuit compilers.

Compares the OLD compiler (``hwman.compiler.qick_codegen``, which generates
Python source and ``exec``s it) against the NEW data-driven compiler
(``hwman.compiler.circuit_program``) and reports whether they produce identical
QICK programs.

It does this WITHOUT any hardware: instead of running the program on a board, it
drives each compiler's ``_initialize``/``_body`` with a recording stand-in passed
as ``self``, capturing the exact ordered sequence of QICK API calls
(``declare_gen``, ``add_gauss``, ``add_pulse``, ``pulse``, ``trigger``, ...) plus
the data-specs attached by the ``@QickBoardSweep`` decorator. If both the call
sequence and the specs match, the resulting sweeps are equivalent.

Usage:
    uv run python scripts/compare_compilers.py
    uv run python scripts/compare_compilers.py -v   # also print full call logs

Note: this is a developer tool, intentionally kept out of the pytest suite.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

import cqedtoolbox.instruments.qick.qick_sweep_v2 as qsv

from lccfq_backend.model.tasks import Gate
from hwman.compiler.circuit import Circuit
from hwman.compiler.circuit_program import CircuitProgram, build_circuit_sweep
from hwman.compiler.qick_codegen import compile_circuit_to_qick


# A representative QICK config. Keys must cover everything both compilers read.
# By default every qubit shares ``q_dac_ch``; per-qubit overrides (``q{N}_dac_ch``)
# can be added per test case to exercise multi-channel behaviour.
BASE_CFG: Dict[str, Any] = {
    "ro_adc_ch": 0,
    "ro_dac_ch": 1,
    "q_dac_ch": 2,
    "q_nqz": 1,
    "ro_nqz": 2,
    "ro_len": 300,
    "ro_freq": 100.0,
    "ro_gain": 0.5,
    "trig_time": 0.5,
    "q_freq": 4000.0,
    "q_pi_sigma": 10,
    "q_pi_n_sigma": 4,
    "q_pi_gain": 0.9,
}

Call = Tuple[str, tuple, dict]


class Recorder:
    """Stand-in for an ``AveragerProgramV2`` that records every API call.

    Passed as ``self`` to the compilers' ``_initialize``/``_body`` so we capture
    the QICK calls they would make, in order, without touching hardware.
    """

    # Attributes the NEW compiler reads off ``self`` (must not be recorded).
    _REAL_ATTRS = {"circuit", "_qubit_gen_ch", "calls"}

    def __init__(self, circuit: Circuit | None = None) -> None:
        self.calls: List[Call] = []
        self.circuit = circuit
        # _qubit_gen_ch is a staticmethod -> plain function; call it for real.
        self._qubit_gen_ch = CircuitProgram._qubit_gen_ch

    def __getattr__(self, name: str):
        # Only reached for attributes not set on the instance, i.e. QICK API
        # methods like declare_gen/add_pulse/pulse/trigger/...
        def recorder(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args, kwargs))

        return recorder


class _IdentityDecorator:
    """Replacement for ``QickBoardSweep`` that captures specs and is a no-op.

    Lets us recover the underlying program class (and the decorator's data-specs)
    from the OLD compiler's ``exec``'d source without building a real Sweep.
    """

    def __init__(self, *specs: Any) -> None:
        self.specs = specs

    def __call__(self, cls: type) -> type:
        cls._captured_specs = self.specs  # type: ignore[attr-defined]
        return cls


def _record_new(circuit: Circuit, cfg: Dict[str, Any]) -> Tuple[List[Call], list]:
    """Capture the NEW compiler's QICK calls and data-specs."""
    rec = Recorder(circuit)
    CircuitProgram._initialize(rec, cfg)
    CircuitProgram._body(rec, cfg)
    specs = list(build_circuit_sweep(circuit).get_data_specs())
    return rec.calls, specs


def _record_old(circuit: Circuit, cfg: Dict[str, Any]) -> Tuple[List[Call], list]:
    """Capture the OLD compiler's QICK calls and data-specs.

    Generates the program source, ``exec``s it with ``QickBoardSweep`` patched to
    an identity decorator (so we get the class, not a Sweep), then drives the
    class methods with a Recorder as ``self``.
    """
    source = compile_circuit_to_qick(circuit, "CompiledProgram")

    saved = qsv.QickBoardSweep
    qsv.QickBoardSweep = _IdentityDecorator  # type: ignore[assignment,misc]
    try:
        ns: Dict[str, Any] = {}
        exec(source, ns)
    finally:
        qsv.QickBoardSweep = saved  # type: ignore[assignment,misc]

    cls = ns["CompiledProgram"]
    specs = list(cls._captured_specs)

    rec = Recorder(circuit)
    cls._initialize(rec, cfg)
    cls._body(rec, cfg)
    return rec.calls, specs


def _normalize(value: Any) -> Any:
    """Round floats so tiny representation noise doesn't show as a difference."""
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (list, tuple)):
        return type(value)(_normalize(v) for v in value)
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _fmt_call(call: Call) -> str:
    name, args, kwargs = call
    parts = [repr(_normalize(a)) for a in args]
    parts += [f"{k}={_normalize(v)!r}" for k, v in kwargs.items()]
    return f"{name}({', '.join(parts)})"


def _fmt_spec(spec: Any) -> str:
    depends = getattr(spec, "depends_on", None)
    return f"{type(spec).__name__}(name={spec.name!r}, depends_on={depends!r})"


def _calls_equal(a: List[Call], b: List[Call]) -> bool:
    return [_normalize(c) for c in a] == [_normalize(c) for c in b]


def _specs_equal(a: list, b: list) -> bool:
    def key(s: Any) -> tuple:
        return (type(s).__name__, s.name, tuple(getattr(s, "depends_on", []) or []))

    return [key(s) for s in a] == [key(s) for s in b]


def _diff_calls(old: List[Call], new: List[Call]) -> List[str]:
    """Align the two call sequences and show insertions/deletions cleanly.

    Uses difflib so that a single inserted/removed call shows as one ``+``/``-``
    line rather than misaligning (and appearing to reorder) everything after it.
    """
    old_fmt = [_fmt_call(c) for c in old]
    new_fmt = [_fmt_call(c) for c in new]
    lines = []
    for line in difflib.unified_diff(old_fmt, new_fmt, fromfile="OLD", tofile="NEW", lineterm="", n=1):
        lines.append(f"    {line}")
    return lines


def _make_cases() -> List[Tuple[str, Circuit, Dict[str, Any]]]:
    """Build the test circuits, each with the cfg it should run against."""

    def measure(q: int) -> Gate:
        return Gate(symbol="measure", target_qubits=[q], control_qubits=[], params=[])

    def g(sym: str, q: int, params: list | None = None) -> Gate:
        return Gate(symbol=sym, target_qubits=[q], control_qubits=[], params=params or [])

    cases: List[Tuple[str, Circuit, Dict[str, Any]]] = []

    cases.append(("X on q0 + measure", Circuit([g("x", 0), measure(0)], 1000, "c1"), BASE_CFG))
    cases.append(("Y on q0 + measure", Circuit([g("y", 0), measure(0)], 500, "c2"), BASE_CFG))
    cases.append(("RX(1.57) on q0 + measure", Circuit([g("rx", 0, [1.57]), measure(0)], 1000, "c3"), BASE_CFG))
    cases.append(("RX(pi) on q0 + measure", Circuit([g("rx", 0, [float(np.pi)]), measure(0)], 1000, "c4"), BASE_CFG))
    cases.append(("RY(0.3) on q0 + measure", Circuit([g("ry", 0, [0.3]), measure(0)], 1000, "c5"), BASE_CFG))

    # Multi-qubit, all sharing the default qubit DAC channel.
    multi = Circuit([g("x", 0), g("x", 1), measure(1), measure(0)], 1000, "c6")
    cases.append(("Multi-qubit (shared channel), reordered measure", multi, BASE_CFG))

    # Multi-qubit with distinct per-qubit DAC channels (multi-channel envelope).
    multi_ch_cfg = {**BASE_CFG, "q0_dac_ch": 2, "q1_dac_ch": 3}
    cases.append(("Multi-qubit, per-qubit channels", multi, multi_ch_cfg))

    # Circuit that never touches qubit 0.
    cases.append(("X on q1 + measure q1 (no qubit 0)", Circuit([g("x", 1), measure(1)], 1000, "c8"), BASE_CFG))

    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="print full call logs")
    args = parser.parse_args()

    cases = _make_cases()
    identical = 0
    differing = 0

    for name, circuit, cfg in cases:
        print("=" * 78)
        print(f"CASE: {name}")
        print("-" * 78)

        try:
            old_calls, old_specs = _record_old(circuit, cfg)
        except Exception as exc:  # noqa: BLE001 - we want to surface any failure
            differing += 1
            print(f"  OLD compiler RAISED {type(exc).__name__}: {exc}")
            new_calls, new_specs = _record_new(circuit, cfg)
            print(f"  NEW compiler succeeded with {len(new_calls)} calls.")
            print("  => DIFFERENT (new compiler handles a case the old one could not)")
            continue

        new_calls, new_specs = _record_new(circuit, cfg)

        calls_ok = _calls_equal(old_calls, new_calls)
        specs_ok = _specs_equal(old_specs, new_specs)

        if args.verbose:
            print(f"  OLD specs: {[_fmt_spec(s) for s in old_specs]}")
            print(f"  NEW specs: {[_fmt_spec(s) for s in new_specs]}")
            print("  OLD calls:")
            for c in old_calls:
                print(f"    {_fmt_call(c)}")
            print("  NEW calls:")
            for c in new_calls:
                print(f"    {_fmt_call(c)}")

        if calls_ok and specs_ok:
            identical += 1
            print(f"  IDENTICAL  ({len(new_calls)} calls, {len(new_specs)} specs)")
        else:
            differing += 1
            print("  DIFFERENT")
            if not specs_ok:
                print("  data-specs differ:")
                print(f"    OLD: {[_fmt_spec(s) for s in old_specs]}")
                print(f"    NEW: {[_fmt_spec(s) for s in new_specs]}")
            if not calls_ok:
                print("  call sequence differs (unified diff, OLD -> NEW):")
                for line in _diff_calls(old_calls, new_calls):
                    print(line)

    print("=" * 78)
    print(f"SUMMARY: {identical} identical, {differing} different (of {len(cases)} cases)")
    print(
        "\nNote: differences are expected only for circuits that exercise the old\n"
        "compiler's known bugs (envelope added on a single hard-coded channel /\n"
        "assuming qubit 0 is present). On the common single-channel path the two\n"
        "compilers are identical."
    )
    return 0 if differing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
