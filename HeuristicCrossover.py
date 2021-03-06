import matplotlib.pyplot as plt
import numpy as np
import itertools
import re
P = 2
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
    perms = list(itertools.permutations(list(np.arange(N)), N))
    parents = []
    for i in range(P):
        parents.append(perms[i])
    return parents

print(generate_permutations(6))




