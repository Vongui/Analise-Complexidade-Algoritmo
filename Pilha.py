pilha = []

# pilha.append(1)
# pilha.append(2)
# print(pilha.pop()) # Remove e retorna o topo da pilha

def push(pilha, item):
    pilha.append(item)

def peek(pilha):
    if pilha:
        return pilha[-1]
    return None

print(pilha)
push(pilha, 'A')
push(pilha, 'B')

print(pilha)
pilha.pop()
print(peek(pilha))


#push() ->
#pop() ->
#