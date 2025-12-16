import qsharp
from QWalkVChain5Microsoft import ExecuteCircuitQSharp

def main():
    shots = 10

    print("Ejecutando circuito Q#...\n")

    for i in range(shots):
        result = ExecuteCircuitQSharp.simulate()
        print(f"Shot {i}: {result}")

if __name__ == "__main__":
    main()
