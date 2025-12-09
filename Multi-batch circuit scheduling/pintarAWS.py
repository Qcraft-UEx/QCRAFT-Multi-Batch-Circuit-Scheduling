from braket.aws import AwsQuantumTask
from braket.circuits import Circuit
#NO FUNCIONA
# Reemplaza con tu ARN real
task = AwsQuantumTask("arn:aws:braket:us-east-1:421778955839:quantum-task/787e2b82-9120-49c5-af83-33d546f97901")

# Recuperamos el OpenQASM de la tarea ejecutada
ir = task.ir

print("\n===== OpenQASM del circuito ejecutado =====\n")
print(ir)

# Reconstruimos el circuito de Braket
circuit = Circuit.from_ir(ir)

# Dibujamos el circuito (modo gráfico)
fig = circuit.draw()

# Guardar como imagen
fig.savefig("circuito_ejecutado.png")
print("\nImagen guardada como circuito_ejecutado.png")
