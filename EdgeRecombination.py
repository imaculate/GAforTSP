import matplotlib.pyplot as plt
import numpy as np
import math
import re
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

def crossover(P1, P2):
    count = 0
    N = len(P1)
    while count < N:
        count+=1


