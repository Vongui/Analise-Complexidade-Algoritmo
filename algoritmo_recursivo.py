import time

def fibonacci_recursivo(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursivo(n - 1) + fibonacci_recursivo(n - 2)

def fibonocci_iterativo(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1): # "_" underscore representa uma variavel na qual nao será utilizada o valor
        a, b = b, a + b
    return b

start_time = time.time()
print(fibonacci_recursivo(30))
final_time = time.time()
print(f"tempo de execucao recursivo: {(final_time-start_time)*100}")

start_time = time.time()
print(fibonocci_iterativo(500))
final_time = time.time()
print(f"tempo de execucao iterativo: {(final_time-start_time)*100}")
#0.11552190780639648