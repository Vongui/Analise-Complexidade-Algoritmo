import math
import time


def fibonacci_binet(n):
    phi = (1 + math.sqrt(5)) / 2
    return round((phi ** n - (1 - phi) ** n) / math.sqrt(5))


start_time = time.time()
result = fibonacci_binet(500)
final_time = time.time()
#print(f"Resultado: {result}" )
print(f"tempo de execucao : {(final_time-start_time)*100}")
#0.000858306884765625