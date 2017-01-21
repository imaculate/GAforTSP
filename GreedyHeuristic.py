import matplotlib.pyplot as plt
import numpy as np
import math
import re
import random
import time

P = 30
problem = 'Data/st70.tsp'

f = open(problem, 'r')
for i in range(6):
     f.readline()
points = []

m = re.search("\d", problem).start()
n = problem.index('.',m)
EPS = 0.00000001
num_points = eval(problem[m:n])
# print(num_points)

for i in range(num_points):
    arr = []
    line = f.readline()
    sarr = line.split(" ")
    c = False
    for s in sarr:
        s = s.strip()
        if(s.isdigit() or s.replace('.','',1).replace('e+','',1).isdigit()):
            if c:
                 arr.append(int(float(s)))
            else:
                c = True
    #print(sarr)
    points.append(arr)


f.close()

points = np.array(points).astype(float)
# plt.plot(points[:,0], points[:,1], 'o')
# plt.show()

def generate_permutations(N):
    perms = []
    counter = 0
    fact = math.factorial(N)

    while  counter < fact and counter < P:
        n = list(np.random.permutation(N))
        if(n not in perms):
            perms.append(n)
            counter+= 1
    return perms




def create_edges(parents):
    rets = []
    for parent in parents:
        ret = [-1] * num_points
        ret[parent[num_points-1]] = parent[0]
        for i in range(num_points-1):
            ret[parent[i]] = parent[i+1]
        rets.append(ret)

    return rets

def encode_edge(parent):
    ret = [-1] * num_points
    ret[parent[num_points-1]] = parent[0]
    for i in range(num_points-1):
        ret[parent[i]] = parent[i+1]
    return ret


def point_included(p, chain):
    if( p not in chain and chain[p]==-1):
        return False
    else:
        return True


def crossover(encoded_parents):
    count = 0
    child = [-1]*num_points

    last = curr = random.randint(0, num_points-1) #random initial start

    while count < num_points-1:
        lens=[ np.linalg.norm(points[curr]-points[x[curr]]) for x in encoded_parents]
        sorted_lens = sorted(lens)

        nxt_parent_index  = lens.index(sorted_lens[0])
        hi = 0
        while(point_included(encoded_parents[nxt_parent_index][curr],child)):
            hi+=1
            nxt_parent_index  = lens.index(sorted_lens[hi])

        nxt = encoded_parents[nxt_parent_index][curr]
        child[curr] = nxt
        curr = nxt

        count+=1
    child[curr] = last
    return child


def chain_length(chain):
    total  = 0
    for i  in range(num_points):
        total+= round(np.linalg.norm(points[i]- points[chain[i]]))

    return total

def decode_edges(chain):
    decoded = []
    curr = 0
    for i in range(num_points):
        decoded.append(points[curr])
        curr = chain[curr]
    return decoded

def insert_mutate(chain):
    decoded = decode_edges(chain)
    rn1 = random.randint(0,num_points-1)
    rn2 = random.randint(0, num_points-1)
    while(rn2 ==rn1):
        rn2 = random.randint(0, num_points-1)
    left = min(rn1, rn2)
    right = min(rn1, rn2)
    tmp = decoded[right]
    for m in range(left+2, right+1):
        decoded[m]=decoded[m-1]
    decoded[left+1] = tmp
    return encode_edge(decoded)

def flip_mutate(child):
    decoded = decode_edges(child)
    rn1 = random.randint(0,num_points-1)
    rn2 = random.randint(0, num_points-1)
    while(rn2 ==rn1):
        rn2 = random.randint(0, num_points-1)
    tmp = decoded[rn1]
    decoded[rn1] = decoded[rn2]
    decoded[rn2] = tmp
    return encode_edge(decoded)

def inverse_mutate(child):
    decoded = decode_edges(child)
    rn1 = random.randint(0,num_points-1)
    rn2 = random.randint(0, num_points-1)
    while(rn2 ==rn1):
        rn2 = random.randint(0, num_points-1)
    left = min(rn1, rn2)
    right = max(rn1, rn2)

    while(right>left):
        tmp = decoded[right]
        decoded[right] = decoded[left]
        decoded[left] = tmp

    return encode_edge(decoded)



def make_mutations(child):
    #not sure if insert, inverse or flip mutation but definitely make more than P childre
    children = [[-1]*num_points]* ((3*P)+1)
    children[0] = child
    for i in range(1,P+1):
        children[i] = insert_mutate(child)
    for i in range(P+1, 2*P+1):
        children[i] = flip_mutate(child)
    for i in range(2*P+1,  3*P+1):
        children[i]= inverse_mutate(child)

    return children

max_iterations = 200
iter = 0

parents = generate_permutations(num_points)
encoded_parents = create_edges(parents)
prev_short = -1

t0 = time.time()
while(iter<max_iterations):
    print("Iteration number  ,", iter+1)
    child = crossover(encoded_parents)
    children = make_mutations(child)
    sorted_children = sorted(children, key = chain_length)
    encoded_parents = sorted_children[:P]
    shortest = chain_length(encoded_parents[0])
    if abs(prev_short- shortest) < EPS:
        break
    prev_short = shortest
    print(shortest)
    iter+=1


winner = encoded_parents[0]
print("Length is , ", chain_length(winner))
t1 = time.time()
print("It took ", t1-t0, " seconds")

dp = np.array(decode_edges(winner))
plt.plot(points[:,0], points[:,1], 'o')
plt.plot(dp[:,0], dp[:,1], 'r--', lw=2)
plt.show()





