# Write a program that, given a number, it controls if it is prime or not

import math

def isPrime(n) -> bool:
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0: return False
    
    sqrt = int(math.isqrt(n))
    
    for i in range(3,sqrt, 2):  # step of two cuz i have already verified if it is a multiple of 2
        if n % i == 0: return False
    return True

def recursiveIsPrime(n, i):
    if n <= 1: return False
    if i == 1: return True
    # if i >= n: return True    | This is another version where we start with i = 2 and increment it
    if n % i == 0: return False
    # recursiveIsPrime(n,i+1)
    return recursiveIsPrime(n, i - 1)

n = 13

if isPrime(n):
    print(f"The number {n} is a prime number")
else:
    print(f"The number {n} is not a prime number")
    
if recursiveIsPrime(n, n-1):
    print(f"The number {n} is a prime number")
else:
    print(f"The number {n} is not a prime number")
