str = "Hello how are you doing?    "

# To remove blank spaces at the end of the string --> .strip()
# To replace every kind of character --> .replace(char to replace, new char)

# ---------------------------------------------------

# first n characters
n = 3
print(str[:n])

# ---------------------------------------------------

# without the first and the last character
print(str.strip()[1:len(str.strip()) - 1])

# ---------------------------------------------------

# only characters in even position
print(str.strip()[0::2])

# ---------------------------------------------------

# reverse string
print(str.strip()[::-1])

# ---------------------------------------------------

# substitute a substring given start and end indices
startIndex = 10
endIndex = 12 # included
print(str.replace(str[startIndex:endIndex+2], "").strip())  # +2, because i want to include the "e" of "are" (+1) and moreover i do not want two consecutive blanks --> +1 --> +2

# ---------------------------------------------------

# rotate to the right
k = 2
print(str.strip()[len(str.strip())-k:]+str.strip()[0:len(str.strip())-k])

# ---------------------------------------------------

# rotate to the left
k = 2
print(str.strip()[k:]+str.strip()[0:k])

# ---------------------------------------------------

# split the string in blocks of length k
def chunk(str, k):
    return [str[i:i+k] for i in range(0, len(str), k)]

k = 2
print(chunk(str.strip(), k))
