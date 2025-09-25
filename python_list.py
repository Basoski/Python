import math
import heapq

def add(a, b):
    if len(a) != len(b):
        raise ValueError("Length of a and b must coincide")
    return [x + y for x, y in zip(a, b)]


def scale(k, a):
    return [k * x for x in a]


def scalar_product(a, b):
    if len(a) != len(b):
        raise ValueError("Length of a and b must coincide")
    res = 0
    for i in range(0, len(a)):
        res += a[i] * b[i]
    return res


def norm1(a):
    res = 0
    for i in range(0, len(a)):
        res += abs(a[i])
    return res


def norm2(a):
    res = 0
    for i in range(0, len(a)):
        res += a[i] ** 2
    return math.sqrt(res)

def normalize(a):
    norm = norm2(a)
    return [x/norm for x in a]

def proj(a,b):
    return scale((scalar_product(a,b) / scalar_product(b,b)), b)

def cumulative_sum(a):
    newVec = []
    currentSum = 0
    for i in range(0,len(a)):
        currentSum += a[i]
        newVec.append(currentSum)
    return newVec

def mean(a):
    sum = 0
    for x in a:
        sum += x
    return sum / len(a)

def moving_avg(a,k):
    newList = [a[i:i+k] for i in range(0,len(a), k)]
    means = []
    for l in newList:
        means.append(mean(l))
    print(f"Windows: {newList}")
    return means
    
def topk_indices(a,k):
    return [i for i,_ in heapq.nlargest(k,enumerate(a), key= lambda t: t[1])] # O(nlogk). It uses heap. Basically enumerate provides a list of tuples (index, value), key says to nlargest to compare elements using the second element of the tuple (hence, the value)
    
def rle(a):
    myDict = {}
    for x in a:
        if x in myDict.keys(): myDict[x] += 1
        else: myDict[x] = 1
    return myDict

a = [1, 6, 3, 7, 5, 2, 4]
b = [8, 9, 10, 11, 12, 13, 14]

print(add(a, b))
print(scale(3, a))
print(scalar_product(a, b))
print(norm1(a))
print(norm2(a))
print(normalize(a))
print(proj(a,b))
print(cumulative_sum(a))
print(moving_avg(a,4))
print(topk_indices(a,2))
print(rle([1,1,1,2,2,3,3,3,3,2]))
