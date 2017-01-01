import matplotlib.pyplot as plt
import numpy as np
import math
import re
import random

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


def encode_edge(parent):
    ret = [-1] * num_points
    ret[parent[num_points-1]] = parent[0]
    for i in range(num_points-1):
        ret[parent[i]] = parent[i+1]
    return ret

def create_edges(parents):
    rets = []
    for parent in parents:
        rets.append(encode_edge(parent))

    return rets




def point_included(p, chain):
    if( p not in chain and chain[p]==-1):
        return False
    else:
        return True


def get_nearest_point(curr, child, P1, P2):
    opt1 = P1
    opt2 = P2
    if(point_included(opt1,child) and point_included(opt2,child)):
        rn = random.randint(0, num_points-1)
        while(point_included(rn,child)):
            rn = random.randint(0, num_points-1)
        return rn

    else:
        if(point_included(opt1,child) or point_included(opt2,child)):
            rn = random.randint(0, num_points-1)
            while(point_included(rn,child)):
                rn = random.randint(0, num_points-1)
            if(point_included(opt1,child)):
                opt1 = rn
            elif(point_included(opt2,child)):
                opt2 = rn

        d1 = np.linalg.norm(points[curr]- points[opt1])
        d2 = np.linalg.norm(points[curr]- points[opt2])

        if(d1<d2):
            nxt = opt1
        else:
            nxt = opt2

        return nxt


def crossover(P1, P2):
    count = 0
    child = [-1]*num_points

    last = curr = random.randint(0, num_points-1) #random initial start
    while count < num_points-1:
        opt1 = P1[curr]
        opt2 = P2[curr]
        if((point_included(opt1,child) and point_included(opt2,child)) or count>=num_points-3):# will have to double check how to check if a point is in the child already
            rn = random.randint(0, num_points-1)
            while(point_included(rn,child)):
                rn = random.randint(0, num_points-1)
            child[curr] = rn
            curr = rn
            count+=1

        else:
            if(point_included(opt1,child) or point_included(opt2,child)):
                rn = random.randint(0, num_points-1)
                while(point_included(rn,child)):
                    rn = random.randint(0, num_points-1)
                if(point_included(opt1,child)):
                    opt1 = rn
                elif(point_included(opt2,child)):
                    opt2 = rn


            child1 = list(child)
            child1[curr] = opt1
            opt_sec1 = P1[opt1]
            opt_sec2  = P2[opt2]
            s1 = get_nearest_point(opt1, child1, opt_sec1, opt_sec2)
            d1 = np.linalg.norm(points[curr]- points[opt1]) + np.linalg.norm(points[s1]- points[opt1])

            child2 = list(child)
            child2[curr] = opt2
            s2 = get_nearest_point(opt2, child2, opt_sec1, opt_sec2)
            d2 = np.linalg.norm(points[curr]- points[opt2]) + np.linalg.norm(points[s2]- points[opt2])

            if(d1<d2):
                nxt = opt1
                next_sec = s1
            else:
                nxt = opt2
                next_sec = s2

            child[curr] = nxt
            child[nxt] = next_sec

            curr  = next_sec
            count+=2



    child[curr] = last

    return child


def chain_length(chain):
    total  = 0
    for i  in range(num_points):
        total+= int(np.linalg.norm(points[i]- points[chain[i]]))
    return total

def chain_length_decoded(chain):
    total  = 0
    for i  in range(num_points):
        total+= int(np.linalg.norm(points[chain[i]]- points[chain[(i+1)%num_points]]))
    return total
def decode_edges(chain):
    decoded = []
    curr = 0
    for i in range(num_points):
        decoded.append(points[curr])
        curr = chain[curr]
    return decoded

def decode_permutation(chain):
    decoded = [-1]*num_points
    curr = 0
    for i in range(num_points):
        decoded[i] = curr
        curr = chain[curr]
    return decoded

def mutate_inverse(parent):
    chain = list(parent)
    # edges = [-1]*num_points
    # for i in range(num_points-1):
    #     edges[i] = int(np.linalg.norm(points[chain[i]]-points[chain[i+1]]))
    # edges[num_points-1]=int(np.linalg.norm(points[chain[num_points-1]]-points[chain[0]]))
    # sorted_edges = sorted(edges)
    # e1 = edges.index(sorted_edges[0])
    # j=1
    # e2 = edges.index(sorted_edges[j])
    # while(abs(e2-e1)<2):
    #     j+=1
    #     e2 = edges.index(sorted_edges[j])
    #
    # left = min(e1, e2)
    # right = max(e1, e2)+1
    # if right==num_points:#shift everyth ing left
    #     chain = chain[1:]+ chain[0]
    #     left-=1
    #     right-=1
    rn1 = random.randint(0, num_points-1)
    rn2 = random.randint(0, num_points-1)
    while(rn1==rn2):
        rn2 = random.randint(0, num_points-1)
    left = min(rn1, rn2)
    right = max(rn1, rn2)
    print("Inverting between ", left, " and ", right)
    while(right>left):
        temp = chain[left]
        chain[left] = chain[right]
        chain[right] = temp
        left+=1
        right-=1
    return chain

def twoOptSwap(tour, i, k):
    dec = k
    for c in range(i,int((k+1+i)/2)):

        tmp = tour[dec]
        tour[dec]= tour[c]
        tour[c] = tmp
        dec-=1



def calc_savings(tour, i, k):
    d1 = int(np.linalg.norm(points[tour[(i-1)%num_points]]-points[tour[k]]) + np.linalg.norm(points[tour[i]]-points[tour[(k+1)%num_points]]))
    d2  =int(np.linalg.norm(points[tour[(i-1)%num_points]]- points[tour[i]]) + np.linalg.norm(points[tour[k]]- points[tour[(k+1)%num_points]]))
    return d1-d2

def twoOpt(tour):
    improve = 0
    while ( improve < 20 ):

        best_distance = chain_length_decoded(tour)

        for i in range(num_points-1):

            for k in range(i+1, num_points):
                if(k==(i-1)%num_points):
                    continue
                savings = calc_savings(tour, i, k)

                if (  savings < 0 ):
                    improve = 0
                    twoOptSwap( tour,i, k )
                    best_distance -= savings
                    print(best_distance)


        improve +=1
    # plt.plot(points[:,0], points[:,1], 'o')
    # plt.plot(tour[:,0], tour[:,1], 'r--', lw=2)
    # plt.show()
    return tour

max_iterations = 50
iter = 0

parents = generate_permutations(num_points)
encoded_parents = create_edges(parents)
prev_short = -1

while(iter<max_iterations):
    print("Iteration number  ,", iter+1)
    children = []
    for i in range(P):
        for j in range(i):
            child = crossover(encoded_parents[i], encoded_parents[j])
            children.append(child)
        for j in range(i+1,P):
            child = crossover(encoded_parents[i], encoded_parents[j])
            children.append(child)

    sorted_children = sorted(children, key = chain_length)
    encoded_parents = sorted_children[:P]
    shortest = chain_length(encoded_parents[0])
    iter+=1
    if (abs(prev_short -shortest)<=EPS):
        print("End of crossover, beginning mutation")
        break
    prev_short = shortest
    print(shortest)


mutes = [encode_edge(twoOpt(decode_permutation(chain))) for chain in encoded_parents]
children  = encoded_parents + mutes
sorted_children = sorted(children, key = chain_length)
encoded_parents = sorted_children[:P]




winner = encoded_parents[0]
print("Length is , ", chain_length(winner))
dp = np.array(decode_edges(winner))

plt.plot(points[:,0], points[:,1], 'o')
plt.plot(dp[:,0], dp[:,1], 'r--', lw=2)
plt.show()

#next idea, do some kind of mutation, or 2-opt, 3-opt



