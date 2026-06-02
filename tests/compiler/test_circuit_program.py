"""
Tests for building QICK programs from Circuit objects.
"""

import numpy as np
import pytest

from lccfq_backend.model.tasks import Gate
from labcore.measurement.sweep import Sweep

from hwman.compiler.circuit import Circuit
from hwman.compiler.circuit_program import (
    build_circuit_sweep,
    measured_qubits_in_order,
    unique_qubits,
    validate,
    _flatten_operations,
)
from hwman.errors import (
    UnsupportedGateError,
    CircuitMissingMeasurementError,
    TwoQubitGateNotImplementedError,
)


def _measure(q: int) -> Gate:
    return Gate(symbol="measure", target_qubits=[q], control_qubits=[], params=[])


def test_build_circuit_sweep_returns_sweep():
    """A valid circuit compiles into a runnable Sweep (no hardware required)."""
    g0 = Gate(symbol="rx", target_qubits=[0], control_qubits=[], params=[np.pi])
    circuit = Circuit(gates=[g0, _measure(0)], shots=1000, pid="test-rx-measure")

    sweep = build_circuit_sweep(circuit)

    assert isinstance(sweep, Sweep)


def test_measurement_order_drives_data_specs():
    """ComplexQICKData specs appear in measurement order, not qubit order."""
    # Measure qubit 2 first, then qubit 0.
    gates = [
        Gate(symbol="x", target_qubits=[0], control_qubits=[], params=[]),
        Gate(symbol="x", target_qubits=[2], control_qubits=[], params=[]),
        _measure(2),
        _measure(0),
    ]
    circuit = Circuit(gates=gates, shots=1000, pid="test-multi-qubit")

    assert measured_qubits_in_order(circuit) == [2, 0]

    sweep = build_circuit_sweep(circuit)
    spec_names = [ds.name for ds in sweep.get_data_specs()]
    assert spec_names == ["repetition", "qubit_2", "qubit_0"]


def test_unique_qubits_sorted():
    gates = [
        Gate(symbol="x", target_qubits=[2], control_qubits=[], params=[]),
        Gate(symbol="x", target_qubits=[0], control_qubits=[], params=[]),
        _measure(2),
    ]
    circuit = Circuit(gates=gates, shots=10, pid="t")
    assert unique_qubits(circuit) == [0, 2]


def test_flatten_operations_pulse_names_are_unique_and_stable():
    """Pulse names are unique and identical across repeated calls (so
    _initialize and _body always agree)."""
    gates = [
        Gate(symbol="x", target_qubits=[0], control_qubits=[], params=[]),
        Gate(symbol="rx", target_qubits=[0], control_qubits=[], params=[1.57]),
        _measure(0),
    ]
    circuit = Circuit(gates=gates, shots=10, pid="t")

    first = [name for _, _, name in _flatten_operations(circuit)]
    second = [name for _, _, name in _flatten_operations(circuit)]

    assert first == second
    assert len(first) == len(set(first))
    assert first == ["x_q0_1", "rx_q0_1p570_2", "measure_q0_3"]


def test_validate_requires_measurement():
    gates = [Gate(symbol="x", target_qubits=[0], control_qubits=[], params=[])]
    circuit = Circuit(gates=gates, shots=10, pid="t")
    with pytest.raises(CircuitMissingMeasurementError):
        validate(circuit)


def test_validate_rejects_unsupported_gate():
    gates = [
        Gate(symbol="cz", target_qubits=[0], control_qubits=[], params=[]),
        _measure(0),
    ]
    circuit = Circuit(gates=gates, shots=10, pid="t")
    with pytest.raises(UnsupportedGateError):
        validate(circuit)


def test_validate_rejects_two_qubit_gate():
    gates = [
        Gate(symbol="x", target_qubits=[1], control_qubits=[0], params=[]),
        _measure(1),
    ]
    circuit = Circuit(gates=gates, shots=10, pid="t")
    with pytest.raises(TwoQubitGateNotImplementedError):
        validate(circuit)


def test_build_circuit_sweep_validates():
    """build_circuit_sweep surfaces validation errors before building."""
    gates = [Gate(symbol="x", target_qubits=[0], control_qubits=[], params=[])]
    circuit = Circuit(gates=gates, shots=10, pid="t")
    with pytest.raises(CircuitMissingMeasurementError):
        build_circuit_sweep(circuit)
