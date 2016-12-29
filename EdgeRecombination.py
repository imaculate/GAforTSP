import matplotlib.pyplot as plt
import numpy as np
import math
import re
import random

P = 10
problem = 'Data/st70.tsp'

f = open(problem, 'r')
for i in range(6):
     f.readline()
points = []

m = re.search("\d", problem).start()
n = problem.index('.',m)

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
plt.plot(points[:,0], points[:,1], 'o')
plt.show()

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


parents = generate_permutations(num_points)
print(parents[0])

def create_edges(ls):
    ret = [0] * num_points
    ret[ls[num_points-1]] = ls[0]
    for i in range(num_points-1):
        ret[ls[i]] = ls[i+1]
    return ret


print(create_edges(parents[0]))

def point_included(p, chain):


def crossover(P1, P2):
    count = 0
    child = [-1]*num_points
    N = len(P1)
    curr = random.randint(0, num_points) #random initial start
    while count < N:
        opt1 = P1[curr]
        opt2 = P2[curr]
        if(opt1 in child and opt2 in child):# will have to double check how to check if a point is in the child already
            rn = random.randint(0, num_points)
            while(rn in child):
                rn = random.randint(0, num_points)
            child[curr] = rn
            curr = rn
            count+=1

        elif(not (opt1 in child or opt2  in child) ):
            d1 = np.linalg.norm(points[curr]- points[opt1])
            d2 = np.linalg.norm(points[curr]- points[opt2])
        else:
            rn = random.randint(0, num_points)
            while(rn in child):
                rn = random.randint(0, num_points)


