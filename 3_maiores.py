import random
import time

start = time.time()
dados = []
for x in range(1000000):
    dados.append(random.randint(1, 100))

dictio = {}
for x in dados:
    if x in dictio.keys():
        dictio[x]+=1
    else:
        dictio[x]=1

dicionario_ordenado = sorted(dictio.items(), key=lambda item: item[1], reverse=True)
end = time.time()
print(f"Tempo de execucao: {end-start} ns")

# print(dicionario_ordenado[0])
# print(dicionario_ordenado[1])
# print(dicionario_ordenado[2])



