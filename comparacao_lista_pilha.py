from collections import deque
import time

def comparar_lista_pilha():
    pilha = []
    start_time = time.time()
    for i in range(500000):
        pilha.append(i)
    while pilha:
        pilha.pop()
    pilha_time = time.time() - start_time

    lista = []
    start_time = time.time()
    for i in range(500000):
        lista.append(i)
    while lista:
        lista.pop(0)
    lista_time = time.time() - start_time

    print(f"Tempo de operação na pilha: {(pilha_time)*1000}")
    print(f"Tempo de operação na lista: {(lista_time)*1000}")

comparar_lista_pilha()