# def encontrar_maior_ineficiente(lista):
#     for i in range(len(lista)):
#         maior = True
#         for j in range(len(lista)):
#             if lista[j] > lista[i]:
#                 maior = False
#                 break
#         if maior:
#             return lista[i]
#
# lista = [5,6,10,22,3,2,1]
# resultado = encontrar_maior_ineficiente(lista)
# print(resultado)
#
# def encontrar_maior_eficiente(lista):
#     maior = lista[0]
#     for i in range(len(lista)):
#         if i > maior:
#             maior = i
#     return maior
#
# resultado = encontrar_maior_eficiente(lista)
# print(resultado)

# def encontrar_ordem_ineficiente(a, b, c):
#     numeros = [a, b, c]
#     for i in range(len(numeros)):
#         maior = True
#         for j in range(len(numeros)):
#             if numeros[j] > numeros[i]:
#                 maior = False
#                 break
#         if maior:
#             maior_num = numeros[i]
#
#
#     for i in range(len(numeros)):
#         menor = True
#         for j in range(len(numeros)):
#             if numeros[j] < numeros[i]:
#                 menor = False
#                 break
#         if menor:
#             menor_num = numeros[i]
#
#     meio_num = sum(numeros) - maior_num - menor_num
#     return maior_num, meio_num, menor_num
#
# def encontrar_ordem_eficiente(a, b, c):
#
#     if a>b and a>c:
#         maior = a
#         meio, menor = c, b
#         if b>c:
#             meio, menor = b, c
#     elif b>a and b>c:
#         maior = b
#         meio, menor = c, a
#         if a>c:
#             meio, menor = a, c
#     else:
#         maior = c
#         meio, menor = b, a
#         if a>b:
#             meio, menor = a,b
#     return maior, meio, menor
#
#
# a, b, c = 20, -1, 1
# maior, meio, menor = encontrar_ordem_ineficiente(a, b, c)
# print("Ineficiente -> Maior:", maior, "Meio:", meio, "Menor:", menor)
# # maior, meio, menor = encontrar_ordem_eficiente(a, b, c)
# # print("Eficiente -> Maior:", maior, "Meio:", meio, "Menor:", menor)
# #
# def mais_frequente_ineficiente(lista):
#     max_contagem = 0
#     mais_frequente = None
#     for i in lista:
#         contagem = 0
#         for j in lista:
#             if i == j:
#                 contagem += 1
#         if contagem > max_contagem:
#             max_contagem = contagem
#             mais_frequente = i
#     return mais_frequente
#
# def mais_frequente_eficiente(lista):
#     dicionario = {}
#     for i in lista:
#         if i in dicionario.keys():
#             dicionario[i] += 1
#         else:
#             dicionario[i] = 1
#     mais_frequente = sorted(dicionario.items(), key=lambda item: item[1], reverse=True)
#     return mais_frequente[0][0]
#
#
# lista_numeros = [3, 1, 4, 1, 5, 9, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 3, 5]
# print("Mais frequente (ineficiente):", mais_frequente_ineficiente(lista_numeros))
# print("Mais frequente (otimizado):", mais_frequente_eficiente(lista_numeros))

def encontrar_pares_ineficiente(lista, alvo):
    pares = []
    for i in range(len(lista)):
        for j in range(i+1, len(lista)):
            if lista[i] + lista[j] == alvo:
                pares.append((lista[i], lista[j]))
    return pares

def encontrar_pares_eficiente(lista, alvo):
    pares = []
    vistos = []
    for i in lista:
        complemento = alvo - i
        if complemento in vistos:
            pares.append((complemento, i))
        vistos.append(i)
    return pares


lista = [1,2,3,4,5,7,8,10,9]
pares = encontrar_pares_eficiente(lista, 6)
print(pares)



# def encontrar_max_min(lista):
#         return lista[0], lista[0]
#     elif len(lista) == 2:
#         return (max(lista), min(lista))
#     else:
#         meio = len(lista) // 2
#         max1, min1 = encontrar_max_min(lista[:meio])
#         max2, min2 = encontrar_max_min(lista[meio:])
#         return max(max1, max2), min(min1, min2)



