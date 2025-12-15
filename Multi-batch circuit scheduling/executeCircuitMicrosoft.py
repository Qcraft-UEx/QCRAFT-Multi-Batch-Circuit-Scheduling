#!/usr/bin/env python
# coding: utf-8

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate
import json
import os
import re
import threading

class executeCircuitMicrosoft:
    def __init__(self):
        self.transpile_lock = threading.Lock()
        self.condition = threading.Condition()
        self.queued_jobs = 0  # Jobs pendientes (simulación local no los usa, pero mantiene la estructura IBM)

    def code_to_circuit(self, code_str: str) -> QuantumCircuit:
        """
        Convierte un string con código Qiskit a QuantumCircuit (similar a IBM parser).
        """
        try:
            lines = code_str.strip().split('\n')
            qreg = creg = circuit = None

            for line in lines:
                if 'import' in line:
                    continue

                if "QuantumRegister" in line:
                    qreg_name = line.split('=')[0].strip()
                    num_qubits = int(line.split('(')[1].split(')')[0].split(',')[0].strip())
                    qreg = QuantumRegister(num_qubits, qreg_name)
                elif "ClassicalRegister" in line:
                    creg_name = line.split('=')[0].strip()
                    num_clbits = int(line.split('(')[1].split(')')[0].split(',')[0].strip())
                    creg = ClassicalRegister(num_clbits, creg_name)
                elif "QuantumCircuit" in line and qreg and creg:
                    circuit = QuantumCircuit(qreg, creg)
                elif "circuit." in line and circuit:
                    if ".c_if(" in line:
                        operation, condition = line.split('.c_if(')
                    else:
                        operation = line
                        condition = None

                    gate_name = operation.split('circuit.')[1].split('(')[0]
                    args = re.split(r'\s*,\s*', operation.split('(', 1)[1].rsplit(')', 1)[0].strip())

                    # Parsing de gates
                    if gate_name == "measure":
                        q_idx = int(re.search(r'\[(\d+)\]', args[0]).group(1))
                        c_idx = int(re.search(r'\[(\d+)\]', args[1]).group(1))
                        circuit.measure(q_idx, c_idx)
                    elif gate_name == "barrier":
                        circuit.barrier()
                    elif gate_name == "reset":
                        q_idx = int(re.search(r'\[(\d+)\]', args[0]).group(1))
                        circuit.reset(q_idx)
                    elif gate_name == "append":
                        gate_type = args[0]
                        qubits = [qreg[int(re.search(r'\[(\d+)\]', arg).group(1))] for arg in args[1:]]
                        control_qubits = qubits[:-1]
                        target_qubit = qubits[-1]
                        if gate_type == 'mc_x_gate':
                            mcx = MCXGate(len(control_qubits))
                            circuit.append(mcx, control_qubits + [target_qubit])
                    else:
                        qubits = [qreg[int(re.search(r'\[(\d+)\]', arg).group(1))] for arg in args if '[' in arg]
                        params = [eval(arg, {"__builtins__": None}) for arg in args if '[' not in arg]
                        if params:
                            getattr(circuit, gate_name)(*params, *qubits)
                        else:
                            getattr(circuit, gate_name)(*qubits)
                    # Se puede añadir condición c_if aquí si es necesario

            if circuit is None and qreg and creg:
                circuit = QuantumCircuit(qreg, creg)

        except Exception as e:
            raise ValueError(f"Error parsing Microsoft circuit: {e}")

        return circuit

    def get_transpiled_circuit_depth_msft(self, circuit: QuantumCircuit) -> int:
        """
        Transpila el circuito localmente para obtener la profundidad (depth).
        """
        with self.transpile_lock:
            qc_basis = transpile(circuit, basis_gates=['u3','cx'], optimization_level=1, backend=AerSimulator())
        return qc_basis.depth()

    def run_save(self, machine: str, circuit: QuantumCircuit, shots: int,
                 users: list, qubit_number: list, circuit_names: list, extra='') -> dict:
        """
        Ejecuta el circuito localmente usando AerSimulator.
        """
        if machine == "local":
            backend = AerSimulator()
            job = backend.run(circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()
        else:
            # Para futuros backends remotos Microsoft/Qiskit
            counts = {}  # placeholder
        return counts
