import time


def fibonocci_iterativo(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1): # "_" underscore representa uma variavel na qual nao será utilizada o valor
        a, b = b, a + b
    return b

start_time = time.time()
print(fibonocci_iterativo(30))
final_time = time.time()
print(f"tempo de execucao: {(final_time-start_time)*100}")
#2.2411346435546875e-05

