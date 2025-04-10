import time
import random

def soma_lista_iterativo(lista):
    soma = 0
    for elemento in lista:
        soma += elemento
    return soma

def soma_lista_recursiva(lista):
    if not lista:
        return 0
    return lista[0] + soma_lista_recursiva(lista[1:])

lista = []
for i in range(500):
   lista.append(random.randint(1, 100))

start_time = time.time()
soma_lista_iterativo(lista)
final_time = time.time()
print(f"tempo de execucao iterativo: {(final_time-start_time)*100}")

start_time = time.time()
soma_lista_recursiva(lista)
final_time = time.time()
print(f"tempo de execucao recursivo: {(final_time-start_time)*100}")