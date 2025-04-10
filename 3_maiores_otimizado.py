import random
from collections import Counter
import time

def top_3_repetidos():
    start_time = time.time()
    lista = [random.randint(0, 100) for _ in range(1000000)]
    contador = Counter(lista)
    mais_comuns = contador.most_common(3)
    resultado = [item[0] for item in mais_comuns]
    end_time = time.time()
    tempo_execucao = end_time - start_time
    return resultado, tempo_execucao

resultados, tempo = top_3_repetidos()
print("Os 3 números mais repetidos são:", resultados)
print("Tempo de execução:", tempo, "segundos")
