from braket.circuits import Circuit
import braket.circuits
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
import time
import os
import json
from braket.aws.aws_quantum_task import AwsQuantumTask
from typing import Optional
import braket
import numpy as np



from braket.circuits import Circuit, Instruction, Gate
from braket.circuits.compiler_directive import CompilerDirective

from qiskit import QuantumCircuit
import re
from braket.circuits import Circuit, Gate, QubitSet

import boto3
import json


BARRIER_QUBIT = 9999  # qubit ficticio reservado para barreras
#EL DE JORGE:
# ============================================================
#  CLEAN PARSER: USER CODE → AWS CIRCUIT
# ============================================================

def code_to_circuit_aws(user_code: str):
    circuit = Circuit()
    max_qubit = -1

    # Normaliza saltos y limpia comentarios
    lines = [ln.strip() for ln in user_code.split("\n") if ln.strip() and not ln.strip().startswith("#")]

    for line in lines:

        # ============================================================
        #  BARRERAS
        # ============================================================
        if re.match(r"^barrier\(\s*\)$", line):
            # barrier()
            circuit.barrier()
            continue

        if match := re.match(r"^barrier\((.+)\)$", line):
            qubits = eval(match.group(1))
            if isinstance(qubits, int): qubits = [qubits]
            circuit.barrier(QubitSet(qubits))
            continue

        # ============================================================
        #  MEASURE
        # ============================================================
        if match := re.match(r"^measure\((.+)\)$", line):
            qubits = eval(match.group(1))
            if isinstance(qubits, int): qubits = [qubits]
            circuit.measure(QubitSet(qubits))
            continue

        # ============================================================
        #  RESET
        # ============================================================
        if match := re.match(r"^reset\((.+)\)$", line):
            qubits = eval(match.group(1))
            if isinstance(qubits, int): qubits = [qubits]
            for q in qubits:
                circuit.reset(q)
            continue

        # ============================================================
        #  SWAP
        # ============================================================
        if match := re.match(r"^swap\((.+),(.+)\)$", line):
            q1 = int(match.group(1))
            q2 = int(match.group(2))
            circuit.swap(q1, q2)
            continue

        # ============================================================
        #  UNARIA (X,H,Z,RY,RX,RZ,...)
        # ============================================================
        if match := re.match(r"^(x|h|z|s|t|y)\((.+)\)$", line, re.IGNORECASE):
            gate = match.group(1).lower()
            q = int(match.group(2))
            getattr(circuit, gate)(q)
            continue

        # Rotaciones
        if match := re.match(r"^r([xyz])\(([^,]+),(.+)\)$", line, re.IGNORECASE):
            axis = match.group(1).lower()
            angle = float(match.group(2))
            q = int(match.group(3))
            getattr(circuit, f"r{axis}")(angle, q)
            continue

        # ============================================================
        #  CNOT
        # ============================================================
        if match := re.match(r"^cnot\((.+),(.+)\)$", line):
            c = int(match.group(1))
            t = int(match.group(2))
            circuit.cnot(c, t)
            continue

        # ============================================================
        #  TOFFOLI (CCNOT)
        # ============================================================
        if match := re.match(r"^ccnot\((.+),(.+),(.+)\)$", line):
            c1 = int(match.group(1))
            c2 = int(match.group(2))
            t = int(match.group(3))
            circuit.ccnot(c1, c2, t)
            continue

        # ============================================================
        #  FUNCIONES AWS NATIVAS (directas)
        # ============================================================
        try:
            if line.startswith("circuit."):
                eval(line, {"circuit": circuit, "Circuit": Circuit, "Gate": Gate})
                continue
        except Exception:
            pass

        print(f"[AVISO] Línea no reconocida y omitida: {line}")

    return circuit



# ============================================================
#  ANALIZADOR DEL CIRCUITO
# ============================================================

def analyze_braket_circuit(circ: Circuit):
    used = sorted({int(q) for instr in circ.instructions for q in instr.target})
    gate_count = {}

    for instr in circ.instructions:
        op = instr.operator.__class__.__name__.lower()
        gate_count[op] = gate_count.get(op, 0) + 1

    return {
        "used_qubits": used,
        "total_instructions": len(circ.instructions),
        "gate_count": gate_count,
        "has_measure": any(instr.operator.__class__.__name__ == "Measure" for instr in circ.instructions),
        "has_reset": any(instr.operator.__class__.__name__ == "Reset" for instr in circ.instructions),
        "has_barrier": any(instr.operator.__class__.__name__ == "Barrier" for instr in circ.instructions),
    }



# ============================================================
#  EXPORTACIÓN A QASM (AWS → QISKIT)
# ============================================================

def braket_to_qiskit(circ: Circuit):
    used = sorted({int(q) for instr in circ.instructions for q in instr.target})
    qc = QuantumCircuit(len(used))

    qmap = {q: i for i, q in enumerate(used)}  # remapping

    for instr in circ.instructions:
        op = instr.operator.__class__.__name__.lower()
        t = [qmap[int(q)] for q in instr.target]

        if op == "h": qc.h(t[0])
        elif op == "x": qc.x(t[0])
        elif op == "z": qc.z(t[0])
        elif op == "y": qc.y(t[0])
        elif op == "s": qc.s(t[0])
        elif op == "t": qc.t(t[0])
        elif op == "cnot": qc.cx(t[0], t[1])
        elif op == "ccnot": qc.ccx(t[0], t[1], t[2])
        elif op == "barrier": qc.barrier(t)
        elif op == "measure": qc.measure(t, t)
        elif op == "swap": qc.swap(t[0], t[1])
        elif op.startswith("r"):  # rotations
            angle = instr.operator.angle
            if op == "rx": qc.rx(angle, t[0])
            if op == "ry": qc.ry(angle, t[0])
            if op == "rz": qc.rz(angle, t[0])

    return qc

#MIO:
def diagram_with_barriers(circuit):
    lines = []
    for instr in circuit.instructions:

        if instr.target == BARRIER_QUBIT:
            lines.append("BARRIER")
            continue

        if instr.operator.name == "Measure":
            qubits = [q.qubit for q in instr.target]
            lines.append(f"MEASURE {qubits}")
            continue

        lines.append(str(instr))
    return "\n".join(lines)






def add_measurements_to_circuit(circuit):
    """
    Agrega mediciones a todos los qubits usados en el circuito.
    Ignora qubits ficticios usados como barreras (ej. 9999).
    """
    all_qubits = set()
    for instr in circuit.instructions:
        if hasattr(instr, "target"):
            # Agregamos solo qubits válidos (no 9999)
            all_qubits.update(q for q in instr.target if q != 9999)

    # Convertimos el set a lista ordenada y medimos todos los qubits
    circuit.measure(list(sorted(all_qubits)))

    return circuit


# def code_to_circuit_aws(code_str: str):
#     """
#     Construye un objeto braket.circuits.Circuit ejecutando el código Python
#     generado por create_circuit(), eliminando importaciones y returns.
#     """

#     # 1. Eliminar imports y returns (NO pueden ir dentro del exec limitado)
#     clean_lines = []
#     for line in code_str.split("\n"):
#         stripped = line.strip()
#         if stripped.startswith("import "): continue
#         if stripped.startswith("from "): continue
#         if stripped.startswith("return"): continue
#         clean_lines.append(line)
#     code_str = "\n".join(clean_lines)

#     # 2. Entorno seguro para exec
#     safe_globals = {
#         "__builtins__": {},  # deshabilita imports
#         "Circuit": braket.circuits.Circuit,
#         "np": np,
#         "pi": np.pi
#     }

#     # Necesario si el usuario no define "circuit = Circuit()"
#     safe_locals = {"circuit": braket.circuits.Circuit()}

#     try:
#         exec(code_str, safe_globals, safe_locals)

#         circuit = safe_locals.get("circuit")
#         if circuit is None:
#             raise ValueError("El código no produjo un objeto 'circuit'.")

#         return circuit

#     except Exception as e:
#         print("\n❌ Error ejecutando el código del circuito AWS:")
#         print(e)
#         print("Código recibido:\n", code_str)
#         raise



def get_transpiled_circuit_depth_aws(circuit:braket.circuits.Circuit, backend) -> None:
    """
    Transpiles a circuit and returns its depth.

    Args:
        circuit (braket.circuits.Circuit): The circuit to transpile.        
        backend (): The machine to transpile the circuit
    """
    # TODO
    return None

def retrieve_result_aws(id:int) -> dict:
    """
    Retrieves the results of a circuit execution from the AWS cloud based on a task id.

    Args:
        id (int): The id of the task to retrieve the results from.
    
    Returns:
        dict: The results of the task execution.
    """
    # Load your AWS account
    task = AwsDevice.retrieve(id)
    return recover_task_result(task).measurement_counts

def recover_task_result(task_load: AwsQuantumTask) -> dict:
    """
    Waits for the task to complete and recovers the results of the circuit execution.

    Args:
        task_load (braket.aws.aws_quantum_task.AwsQuantumTask): The task to recover the results from.
    
    Returns:
        dict: The results of the circuit execution.
    """
    # recover task
    sleep_times = 0
    while sleep_times < 100000:
        status = task_load.state()
        print('Status of (reconstructed) task:', status)
        print('\n')
        # wait for job to complete
        # terminal_states = ['COMPLETED', 'FAILED', 'CANCELLED']
        if status == 'COMPLETED':
            # get results
            return task_load.result()
        else:
            time.sleep(1)
            sleep_times = sleep_times + 1
    print("Quantum execution time exceded")
    return None

def runAWS(machine:str, circuit:Circuit, shots:int, s3_folder: Optional[str] = None) -> dict:
    """
    Executes a circuit in the AWS cloud.

    Args:
        machine (str): The machine to execute the circuit.        
        circuit (Circuit): The circuit to execute.        
        shots (int): The number of shots to execute the circuit.        
        s3_folder (str, optional): The name of the S3 bucket to store the results. Only needed when `machine` is not 'local'
    
    Returns:
        dict: The results of the circuit execution.
    """
    x = int(shots)

    if machine=="local":
        device = LocalSimulator()
        result = device.run(circuit, shots=x).result()
        counts = result.measurement_counts
        print(counts)
        return counts
        
    device = AwsDevice(machine)

    if "sv1" not in machine and "tn1" not in machine:

        s3_folder = ('amazon-braket-jorgecs', 'test/') #TODO change this

        task = device.run(circuit, s3_folder, shots=x, poll_timeout_seconds=5 * 24 * 60 * 60) # Hacer lo mismo que con ibm para recuperar los resultados, guardar el id, usuarios... y despues en el scheduler, al iniciarlo, buscar el el bucket s3 si están los resultados, si no, esperar a que lleguen
        counts = recover_task_result(task).measurement_counts
        return counts
    else:
        task = device.run(circuit, s3_folder, shots=x)
        counts = task.result().measurement_counts
        return counts
    

def runAWS_save(machine:str, circuit:Circuit, shots:int, users:list, qubit_number:list, circuit_names:list, s3_folder: Optional[str] = None) -> dict:
    """
    Executes a circuit in the AWS cloud and saves the task id if the machine crashes.

    Args:
        machine (str): The machine to execute the circuit.        
        circuit (Circuit): The circuit to execute.
        shots (int): The number of shots to execute the circuit.        
        users (list): The users that executed the circuit.        
        qubit_number (list): The number of qubits of the circuit per user.
        circuit_names (list): The name of the circuit that was executed per user.        
        s3_folder (str, optional): The name of the S3 bucket to store the results. Only needed when `machine` is not 'local'

    Returns:
        dict: The results of the circuit execution.
    """
    x = int(shots)

    if machine=="local":
        device = LocalSimulator()
        result = device.run(circuit, shots=x).result()
        counts = result.measurement_counts
        print(counts)
        return counts
        
    device = AwsDevice(machine)

    if "sv1" not in machine and "tn1" not in machine:

        s3_folder = ('amazon-braket-jorgecs', 'Test Circuits/')  # Correct format #TODO change this

        task = device.run(circuit, s3_folder, shots=x, poll_timeout_seconds=5 * 24 * 60 * 60) # Hacer lo mismo que con ibm para recuperar los resultados, guardar el id, usuarios... y despues en el scheduler, al iniciarlo, buscar el el bucket s3 si están los resultados, si no, esperar a que lleguen

        #------------------------#
        id = task # Get the job id
        user_shots = [shots] * len(circuit_names)
        provider = 'aws'
        script_dir = os.path.dirname(os.path.realpath(__file__))
        ids_file = os.path.join(script_dir, 'ids.txt')  # create the path to the results file in the script's directory
        with open(ids_file, 'a') as file:
            file.write(json.dumps({id:(users,qubit_number)}))
            file.write(json.dumps({id:(users,qubit_number, user_shots, provider, circuit_names)}))
            file.write('\n')
        #------------------------#

        counts = recover_task_result(task).measurement_counts

        #------------------------#
        with open(ids_file, 'r') as file:
            lines = file.readlines()
        with open(ids_file, 'w') as file:
            for line in lines:
                line_dict = json.loads(line.strip())
                if list(line_dict.keys())[0] != id:
                    file.write(line)
        #------------------------#

        return counts
    else:
        task = device.run(circuit, s3_folder, shots=x)
        counts = task.result().measurement_counts
        return counts



""""
A partir de aqui todo lo mio
"""

def AWS():
    regiones = ["us-west-1", "us-west-2", "us-east-1", "eu-north-1"]
    dispositivos = []
    
    for region in regiones:
        try:
            client = boto3.client("braket", region_name=region)
            response = client.search_devices(filters=[])
            
            if "devices" in response:
                for device in response["devices"]:
                    device_arn = device.get("deviceArn", "N/A")
                    device_details = client.get_device(deviceArn=device_arn)
                    device_status = device_details.get("deviceStatus", "N/A")

                    if device_status == "RETIRED" or device_status == "OFFLINE":
                        continue  # Saltar dispositivos retirados

                    capabilities = json.loads(device_details.get("deviceCapabilities", "{}"))
                    paradigm = capabilities.get("paradigm", {})

                    if "nativeGateSet" in paradigm:
                        qubit_count = paradigm.get("qubitCount", "N/A")

                        # Obtener el tamaño de la cola de tareas cuánticas
                        queue_info = device_details.get("deviceQueueInfo", [])
                        queue_size = min(
                            [int(q.get("queueSize", float("inf"))) for q in queue_info],
                            default=float("inf")
                        )

                        device_info = {
                            "region": region,
                            "deviceArn": device_arn,
                            "deviceName": device.get("deviceName", "N/A"),
                            "queueSize": queue_size if queue_size != float("inf") else 0,
                            "deviceStatus": device_status,
                            "deviceType": device.get("deviceType", "N/A"),
                            "providerName": device.get("providerName", "N/A"),
                            "qubitCount": qubit_count,
                        }

                        dispositivos.append(device_info)
        
        except Exception as e:
            print(json.dumps({"error": str(e), "region": region}))

    # ✅ Imprimir y devolver la lista de dispositivos
    #print(json.dumps(dispositivos, indent=4))
    return dispositivos  # ✅ Devuelve la lista para su uso posterior