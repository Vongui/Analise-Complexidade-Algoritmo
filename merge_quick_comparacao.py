def merge_sort_visual(arr, depth=0):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        print(f"{'  ' * depth}Dividindo: {arr} -> {left_half} | {right_half}")

        merge_sort_visual(left_half, depth + 1)
        merge_sort_visual(right_half, depth + 1)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

        print(f"{'  ' * depth}Mesclando: {arr}")

# Lista para testar
arr = [8, 3, 5, 2, 7, 6,10,20,99,67,48,78]
print("Lista original:", arr)
merge_sort_visual(arr)

import matplotlib.pyplot as plt
import time


def plot_array(arr, pivot_index=None, title=""):
    plt.figure(figsize=(8, 4))
    colors = ['blue' if i != pivot_index else 'red' for i in range(len(arr))]
    plt.bar(range(len(arr)), arr, color=colors)
    plt.title(title)
    plt.show()
    time.sleep(1)


def quick_sort_visual(arr, left=0, right=None, depth=0):
    if right is None:
        right = len(arr) - 1

    if left < right:
        pivot_index = partition(arr, left, right)

        # Capturando as listas antes e depois do pivô
        left_part = arr[left:pivot_index]
        right_part = arr[pivot_index + 1:right + 1]

        print(f"{'  ' * depth}Passo {depth + 1}: Pivô = {arr[pivot_index]}")
        print(f"{'  ' * depth}  Antes do pivô: {left_part}")
        print(f"{'  ' * depth}  Depois do pivô: {right_part}")
        print(f"{'  ' * depth}  Lista atual: {arr}")

        plot_array(arr, pivot_index, f"Pivot: {arr[pivot_index]} - Particionando")

        quick_sort_visual(arr, left, pivot_index - 1, depth + 1)
        quick_sort_visual(arr, pivot_index + 1, right, depth + 1)


def partition(arr, left, right):
    pivot = arr[right]  # Escolhendo o último elemento como pivô
    i = left - 1

    for j in range(left, right):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1


# Lista para teste
arr = [8, 3, 5, 2, 7, 6]
print("Lista original:", arr)
plot_array(arr, title="Lista Original")

quick_sort_visual(arr)
plot_array(arr, title="Lista Ordenada")

print("Lista ordenada:", arr)





print("Lista ordenada:", arr)
