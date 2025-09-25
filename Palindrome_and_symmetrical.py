# Write a program that verify whether a string is palindrome and/or symmetrical

def reverseString(str):
    # reversed = ""
    # for i in range(len(str) - 1, -1, -1):   # range(start (included), stop (excluded), step)
    #     reversed += str[i]
    # return reversed
    
    # That was C like
    return str[::-1]

def recursiveReverseString(str, reversed, currLen):
    if currLen < 0: return reversed
    reversed += str[currLen]
    return recursiveReverseString(str, reversed, currLen - 1)

def isPalindrome(str):
    if str == reverseString(str): return True
    return False

def isSymmetrical(str):
    if len(str) % 2 != 0: return False
    half = int(len(str) / 2)
    
    for i in range(0,half):
        if str[i] != str[i + half]: return False
    return True


str = "abba"

if isPalindrome(str.lower()):
    print(f"The string {str} is palindrome")
else:
    print(f"The string {str} is not palindrome")
    
if isSymmetrical(str.lower()):
    print(f"The string {str} is symmetrical")
else:
    print(f"The string {str} is not symmetrical")
    