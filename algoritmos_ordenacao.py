import random
import time

def bubble_sort(arr):
    n = len(arr)
    for i in range(n-1):
        for j in range(n-1-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def selection_sort(arr):
    n = len(arr)
    for i in range(n-1):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        chave = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > chave:
            arr[j + 1] = arr[j]
            j -=1
        arr[j + 1] = chave
    return arr

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    meio = len(arr) // 2

    esquerda = arr[:meio]
    direita = arr[meio:]

    esquerda = merge_sort(esquerda)
    direita = merge_sort(direita)

    return merge(esquerda, direita)


def merge(esquerda, direita):
    resultado = []
    i, j = 0, 0

    while i < len(esquerda) and j < len(direita):
        if esquerda[i] <= direita[j]:
            resultado.append(esquerda[i])
            i += 1
        else:
            resultado.append(direita[j])
            j += 1

    resultado.extend(esquerda[i:])
    resultado.extend(direita[j:])

    return resultado

def quick_sort(arr, fim=None, inicio=0):
    if fim is None:
        fim = len(arr) - 1

    if inicio < fim:
        # pivo = partition(arr, inicio, fim)
        pivo = random.randrange(len(arr))
        quick_sort(arr, inicio, pivo - 1)
        quick_sort(arr, pivo + 1, fim)

    return arr

def partition(lista, inicio, fim):
    pivo = lista[inicio]
    anterior = inicio + 1
    posterior = fim

    while True:
        while anterior <= posterior and lista[anterior] <= pivo:
                anterior += 1
        while anterior <= posterior and lista[posterior] > pivo:
            posterior -= 1

        if anterior <= posterior:
            lista[anterior], lista[posterior] = lista[posterior], lista[anterior]
        else:
            lista[inicio], lista[posterior] = lista[posterior], lista[inicio]
            return posterior


#10 elementos
print(f"\n10 elementos ------------------------------------------------------------")
arr = []
for i in range(10):
    x = random.randint(1,100)
    arr.append(x)

# Bubble Sort
start_time = time.time()
bubble = bubble_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Bubble Sort: {final_time - start_time:.6f} segundos")

# Selection Sort
start_time = time.time()
selection = selection_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Selection Sort: {selection}")
print(f"Tempo de execução do Selection Sort: {final_time - start_time:.6f} segundos")

# Insertion Sort
start_time = time.time()
insertion = insertion_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Insertion Sort: {insertion}")
print(f"Tempo de execução do Insertion Sort: {final_time - start_time:.6f} segundos")

#Sorted
start_time = time.time()
sortd = sorted(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Insertion Sort: {insertion}")
print(f"Tempo de execução do Sorted: {final_time - start_time:.6f} segundos")

# Merge sort
start_time = time.time()
merged = merge_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Merge Sort: {final_time - start_time:.6f} segundos")

# Quick Sort
start_time = time.time()
quick = quick_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Quick Sort: {final_time - start_time:.6f} segundos")

#100 elementos
print(f"\n100 elementos ------------------------------------------------------------")
arr = []
for i in range(100):
    x = random.randint(1,100)
    arr.append(x)

# print(f"Lista desordenada: {arr}")

# Bubble Sort
start_time = time.time()
bubble = bubble_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Bubble Sort: {final_time - start_time:.6f} segundos")

# Selection Sort
start_time = time.time()
selection = selection_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Selection Sort: {selection}")
print(f"Tempo de execução do Selection Sort: {final_time - start_time:.6f} segundos")

# Insertion Sort
start_time = time.time()
insertion = insertion_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Insertion Sort: {insertion}")
print(f"Tempo de execução do Insertion Sort: {final_time - start_time:.6f} segundos")

#Sorted
start_time = time.time()
sortd = sorted(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Insertion Sort: {insertion}")
print(f"Tempo de execução do Sorted: {final_time - start_time:.6f} segundos")

# Merge sort
start_time = time.time()
merged = merge_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Merge Sort: {final_time - start_time:.6f} segundos")

# Quick Sort
start_time = time.time()
quick = quick_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Quick Sort: {final_time - start_time:.6f} segundos")

print(f"\n1000 elementos ------------------------------------------------------------")
arr = []
for i in range(1000):
    x = random.randint(1,100)
    arr.append(x)

# print(f"Lista desordenada: {arr}")

# Bubble Sort
start_time = time.time()
bubble = bubble_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Bubble Sort: {final_time - start_time:.6f} segundos")

# Selection Sort
start_time = time.time()
selection = selection_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Selection Sort: {selection}")
print(f"Tempo de execução do Selection Sort: {final_time - start_time:.6f} segundos")

# Insertion Sort
start_time = time.time()
insertion = insertion_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Insertion Sort: {insertion}")
print(f"Tempo de execução do Insertion Sort: {final_time - start_time:.6f} segundos")

#Sorted
start_time = time.time()
sortd = sorted(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Insertion Sort: {insertion}")
print(f"Tempo de execução do Sorted: {final_time - start_time:.6f} segundos")

# Merge sort
start_time = time.time()
merged = merge_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Merge Sort: {final_time - start_time:.6f} segundos")

# Quick Sort
start_time = time.time()
quick = quick_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Quick Sort: {final_time - start_time:.6f} segundos")


print(f"\n10000 elementos ------------------------------------------------------------")
arr = []
for i in range(10000):
    x = random.randint(1,100)
    arr.append(x)

# print(f"Lista desordenada: {arr}")

# Bubble Sort
start_time = time.time()
bubble = bubble_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Bubble Sort: {final_time - start_time:.6f} segundos")

# Selection Sort
start_time = time.time()
selection = selection_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Selection Sort: {selection}")
print(f"Tempo de execução do Selection Sort: {final_time - start_time:.6f} segundos")

# Insertion Sort
start_time = time.time()
insertion = insertion_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Insertion Sort: {insertion}")
print(f"Tempo de execução do Insertion Sort: {final_time - start_time:.6f} segundos")

#Sorted
start_time = time.time()
sortd = sorted(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Insertion Sort: {insertion}")
print(f"Tempo de execução do Sorted: {final_time - start_time:.6f} segundos")

# Merge sort
start_time = time.time()
merged = merge_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Merge Sort: {final_time - start_time:.6f} segundos")

# Quick Sort
start_time = time.time()
quick = quick_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Quick Sort: {final_time - start_time:.6f} segundos")

print(f"\n100000 elementos ------------------------------------------------------------")
arr = []
for i in range(100000):
    x = random.randint(1,100)
    arr.append(x)

# print(f"Lista desordenada: {arr}")

# Bubble Sort
start_time = time.time()
bubble = bubble_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Bubble Sort: {final_time - start_time:.6f} segundos")

# Selection Sort
start_time = time.time()
selection = selection_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Selection Sort: {selection}")
print(f"Tempo de execução do Selection Sort: {final_time - start_time:.6f} segundos")

# Insertion Sort
start_time = time.time()
insertion = insertion_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Insertion Sort: {insertion}")
print(f"Tempo de execução do Insertion Sort: {final_time - start_time:.6f} segundos")

#Sorted
start_time = time.time()
sortd = sorted(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Insertion Sort: {insertion}")
print(f"Tempo de execução do Sorted: {final_time - start_time:.6f} segundos")

# Merge sort
start_time = time.time()
merged = merge_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Merge Sort: {final_time - start_time:.6f} segundos")

# Quick Sort
start_time = time.time()
quick = quick_sort(arr.copy())
final_time = time.time()
# print(f"Lista ordenada pelo Bubble Sort: {bubble}")
print(f"Tempo de execução do Quick Sort: {final_time - start_time:.6f} segundos")