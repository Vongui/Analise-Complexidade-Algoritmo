import time

def fatorial_iterativo(n):
    resultado = 1
    for i in range(2, n + 1):
        resultado *= i
    return resultado

def fatorial_recursivo(n):
    if n == 0 or n == 1:
        return 1
    return n * fatorial_recursivo(n - 1)

start_time = time.time()
fatorial_iterativo(900)
final_time = time.time()
print(f"tempo de execucao iterativo: {(final_time-start_time)*100}")

start_time = time.time()
fatorial_recursivo(900)
final_time = time.time()
print(f"tempo de execucao recursivo: {(final_time-start_time)*100}")