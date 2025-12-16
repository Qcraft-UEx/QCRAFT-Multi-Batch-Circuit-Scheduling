# executeCircuitQSharp.py
import qsharp
from qsharp import Result
from typing import List

# Recarga todos los archivos Q# automáticamente
qsharp.reload_files()

class ExecuteCircuitQSharp:
    def __init__(self):
        pass

    def run_circuit(self) -> List[int]:
        # Llamada directa usando el namespace completo
        # NOTA: Usa qsharp.<Namespace>.<Operation>
        results = qsharp.QcraftExample.QWalkVChain5Microsoft.simulate()
        
        # Convierte Result[] de Q# a lista de 0/1 en Python
        python_results = [0 if r == Result.Zero else 1 for r in results]
        return python_results

if __name__ == "__main__":
    executor = ExecuteCircuitQSharp()
    output = executor.run_circuit()
    print("Resultados de la medición:", output)
