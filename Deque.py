from collections import deque

fila = deque()
for i in range(10):
    fila.append(i)

print(fila)
fila.popleft()
fila.append(11)
print(fila)
fila.popleft()
print(fila)