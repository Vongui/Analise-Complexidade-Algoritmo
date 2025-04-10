import time

def soma_recursiva(n):
    if n == 1:
        return 1
    return n + soma_recursiva(n - 1)

def soma_iterativa(n):
    soma = 0
    for i in range(1, n + 1):
        soma += i
    return soma

start_time = time.time()
soma_iterativa(30)
final_time = time.time()
print(f"tempo de execucao: {(final_time-start_time)*100}")

start_time = time.time()
soma_recursiva(30)
final_time = time.time()
print(f"tempo de execucao: {(final_time-start_time)*100}")

