import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

# NumPy arrays are stored at one continuous place in 
# memory unlike lists, so processes can access and 
# manipulate them very efficiently.
# This behavior is called locality of reference
# This is the main reason why NumPy is faster than lists.

# ------------------ BASIC -----------------------

arr0d = np.array(42) # a scalar
arr1d = np.array([1,2,3,4,5]) # uni-dimensional array --> bunch of 0-D arrays (1x5)
print(arr1d)

arr2d =np.array([[1,2,3] , [4,5,6]]) # bi-dimensional array --> bunch of 1-D arrays (2x3)
print(arr2d)

arr3d = np.array([[[1,2,3] , [4,5,6]], [[7,8,9] , [10,11,12]]])  # three-dimensional array --> bunch of 2-D arrays (2X(2x3)). And so on
print(arr3d)

# to print the dimensions of the arrays: a.ndim
print(arr3d.ndim)

# you can force the number of dimension by doing this
arr = np.array([1,2,3,4,5], ndmin = 5)
print(f"{arr}  {arr.ndim}")

# you can access to any element of the array using the ordinary index system
arr = np.array([1,2,3,4,5])
print(arr[3])

# same thing for any element of multi-dimensional arrays. I.e: for 2-D arrays
print(arr2d[0,1])   # this would access to the second element of the first row
print(arr3d[0,1,2]) # this would access to the third element of the second row of the first matrix 

# also the python indexing remains valid, for example:
print(arr2d[1,-1])    # this would access to the last element of the second row 

# slicing is still valid
arr = np.array([1,2,3,4,5,6,7])
print(arr[1:5]) # indices belonging to [1,5)
print(arr[4:])  # indices belongin to [4, end]
print(arr[:4])  # indices belonging to [start, 4)
print(arr[-3:-1])   # we use the minus operator to refer to an index from the end
print(arr[1:5:2])   # [1,5) with a step of 2 --> 1 + 2 --> 3 + 2
print(arr[::2])    

# this is also valid for multidimensional arrays
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(arr[1,1:4])   # taking the second row, i want to access to the range of indices [1,4)
print(arr[0:2,2])   # from both rows, i access to the elements with index 2
print(arr[0:2, 2:4])

# numpy has data types and refer to them with one character, i.e: i for integers, u for unsigned integers etc.
# by default the type is int32
arr = np.array([1,2,3,4])
print(arr.dtype)

arr = np.array(["cherry","apple","banana"])
print(arr.dtype)    # U6 --> Unicode string

# We can force an array to be a specific type
arr = np.array([1,2,3,4,5], dtype = "S") # S equals to string
print(arr.dtype)

#We can also set the dimensions, i.e:
arr = np.array([1,2,3,4,5], dtype = 'd')
print(arr.dtype) # float64

# We can also convert data type on existing arrays.
# The best way to do it is to make a COPY of the array with
#the astype() method

arr = np.array([1.1,2.1,3.1])
newarr = arr.astype('i') # we can also use the data types of python, example: int (equivalent of int64 for numpy)
print(newarr)

arr = np.array([1, 0, 3])
newarr = arr.astype(bool)   # != 0 --> True, == 0 --> False
print(newarr)


# Copy vs View
# The copy owns the data and any changes made to the copy will not affect the original array. It is also valid for the other way 
# The view does not own the data and any changes made to the view will affect the original array. It is also valid for the other way 

arr = np.array([1,2,3,4])
arr_copy = arr.copy()
arr[0] = 42
print(arr)
print(arr_copy)

arr = np.array([1,2,3,4])
arr_view = arr.view()
arr[0] = -2
print(arr)
print(arr_view)



# Other than the dimensions we can also check the shape of arrays. 
# The shape of an array is the number of elements in each dimension

arr = np.array([1,2,3,4,5])
print(arr.shape)

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(arr.shape)

arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print(arr.shape)


# Since we cna visualize the shape of an array, we can also reshape it

arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newArr = arr.reshape(4,3)
newArr[2,1] = -1
print(newArr)
print(arr)
print(newArr.base)  # it prints the original array (hence it is as view). The property base will print "None" if the variable overwhich "base" is called owns the data
# note how  the reshape method is a view of the original array

newArr = arr.reshape(2,3,2)
print(newArr)

# Note that the elements must be equal in both shapes.
# I.e: we can reshape an 8 elements 1D array into 4 elements in 2 rows 2D array but we cannot
# reshape it into a 3 elements 3 rows 2D array (cuz 3x3 = 9 != 8)

# The reshape method allows us to have one "unkown" dimension. Meaning that if we pass "-1", numpy will calculate this number for you

arr = np.array([1,2,3,4,5,6,7,8])
newArr = arr.reshape(2,2,-1) # this should be (2,2,2) cuz 2x2x2 = 8
print(newArr)

# if we pass only -1 to the reshape method, it will transform every array into a 1D array
arr = np.array([[1,2,3,4] , [5,6,7,8]])
print(arr.shape)
print(arr)
newArr = arr.reshape(-1)
print(newArr.shape)
print(newArr)


# of course we can also iterate over numpy arrays. This can be done with the 
# ordinary python for loops, i.e: for x in 1Darray
# The problem is that, to iterate through each scalar of an array we need to use n for loops for nD arrays. Therefore we have a method that helps us.
# It is very useful for any array with high dimensions
# This method will iterate over each scalar of an array!
arr = np.array([[[1,2] , [3,4]], [[5,6], [7,8]]])
for x in np.nditer(arr):
    print(x)
    
# we can also slice the array passed to this method
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for x in np.nditer(arr[:, 1:4]):
    print(x)
    
    
# sometimes we require corresponding index of teh element while iterating. 
# Indeed, like the enumerate method of python, we have ndenumerate()

arr = np.array([1,2,3])
for idx,x in np.ndenumerate(arr):
    print(idx,x)
    
arr = np.array([[1,2,3,4], [5,6,7,8]])
for idx,x in np.ndenumerate(arr):
    print(idx,x)
    
    
# we can also concatenate arrays.

arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr = np.concatenate((arr1,arr2))   # if not explicitly passed, this method will join them along the axis = 0
print(arr)
print(arr.base) # since it prints out "None". The method concatenate provides a copy, hence a variable that owns the data


arr1 = np.array([[1,2], [3,4]])
arr2 = np.array([[5,6], [7,8]])
arr = np.concatenate((arr1,arr2), axis = 1)
print(arr)


# There is also the method .stack(). It is the same as concatenation, with the difference that stacking is done along a new axis.

arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr = np.stack((arr1,arr2)) # also here, if not passed --> axis = 0
print(arr)
print(arr.base) # also this method provides a copy, hence a variable that owns the data

arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr = np.stack((arr1,arr2), axis = 1)
print(arr)

#THere are also vstack (which is the equivalent to stack((x,y), axis = 0))
# hstack will stack along rows
# dstack will stack along height, which is the same as depth
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.dstack((arr1, arr2))
print(arr)


# Since we can join arrays, we can also split them.
# The method will return a list containing three arrays

arr = np.array([1,2,3,4,5,6])
newArr = np.array_split(arr,3)  # array_split(array to split, k parts)
print(newArr)
print(newArr[0])
print(newArr[0].base) # since it prints the original array, the method array_split provides a view of the original array. Hence the splitted arrays won't own the data
print(newArr[1])
print(newArr[2])

# Valid also for multi dimensional arrays
arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
newArr = np.array_split(arr,3)  # it splits the original array into k 2-D arrays
print(newArr)
print(newArr[0])

# Note that you can also specify along which axis to split
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=1)
print(newarr)


# For hstack, vstack, dstack there are their opposite: hsplit, vsplit, dsplit


# It is possible to search an array for a certain value, and return the indexes that get a match.

arr = np.array([1,2,3,4,5,4,4])
x = np.where(arr == 4) # the method will return an array containing every indexes where the value of the original array is 4
print(x)

# There is a method called searchSorted() which performs a binary search in the array, and retursn the index where the specified value would be inserted to mantain the search order.
# This method is assumed to be used on sorted arrays only

arr = np.array([6,7,8,9])
x = np.searchsorted(arr,4)
print(x) 

# we can tell this method from where to start searching

arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7, side='right')
print(x)

# we can also use this method for multiple values
arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, [1,2,10])  # in this case x will be a list of indexes
print(x)

# we can also sort arrays

arr = np.array([3,2,0,1])
sortedArr = np.sort(arr)
print(sortedArr)
print(sortedArr.base)   # "None" --> sort returns a copy. It owns the data

arr = np.array([[3, 2, 4], [5, 0, 1]])
print(np.sort(arr)) # on multi dimensional arrays it sorts every dimension


# In order to filter arrays we can use boolean lists.
# If the value at an index is true that element is contained in the filtered array, otherwise not

arr = np.array([1,2,3,4])
filterList = [True, True, False, True]
x = arr[filterList]
print(x)
print(x.base) # "None", x owns the data 


arr = np.array([41, 42, 43, 44])
filter_arr = arr > 42
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

# ------------------ BASIC -----------------------

# ------------------ RANDOM -----------------------

# Pseudo randoms --> not truly random numbers because there is an algorithm behind it
# True randoms --> we need to get random data from some outside source, i.e: keystrokes, mouse movements, etc.

x = random.randint(100) # random integer from 0 to 100
print(x)

x = random.rand() # random float from 0 to 1
print(x)

randArr = random.randint(100, size=5)
print(randArr)

randArr2D = random.randint(100, size=(3,5))
print(randArr2D)

randArr = random.rand(5)    # randArr will contain 5 floats (each one from 0 to 1)
print(randArr)

randArr2D = random.rand(3,5)
print(randArr2D)

# we can also randomly pick an element based on an array of values
x = random.choice([3,5,7,9])
print(x)

x = random.choice([3,5,7,9], size=(3,5))
print(x)

# with this in mind, we are able to build a data distribution.
# More specifically we can build a random distribution. This is a set of random numbers that follow a certain probability density function
# Remember: Probability density function: function that describes a continous probability

x = random.choice([3,5,7,9], p=[0.1, 0.3, 0.6, 0.0], size=10) # this will produce a 100 elements 1D array. The elements will be chosen based on the probability given by the paramter p (thus, the value 9 will never occur). Of course the sum of all probabilities should be 1
print(x)


# Numpy random module provides two methods:
# - shuffle: it changes the arrangement of elements in-place (i.e: in the array itself)
# - permutation: it does the same thing, but it returns a re-arranged array (thus, it leaves the original array unchanged)

arr = np.array([1,2,3,4,5])
random.shuffle(arr)
print(arr)

arr = np.array([1,2,3,4,5])
newArr = random.permutation(arr)
print(arr)
print(newArr)


# Visualizing distributions with seaborn.
# Seaborn is a library that uses matplotlib underneath to plot graphs.
# displot stands for distribution plot

#sns.displot([0,1,2,3,4,5])  # histograms
#plt.show()

#sns.displot([0,1,2,3,4,5], kind="kde")  # without histograms
#plt.show()

# in order to build the normal distribution, numpy provides the normal method. It has three parameters:
# - loc: where the peak of the bell exists (mean)
# - scale: how flat the graph fistribution should be (standard deviation)
# - size: the shape of the returned array

x = random.normal(loc = 1, scale = 2, size = (2,3))
#sns.displot(x, kind="kde")
#plt.show()

# it provides also a method for the binomial distribution (it's a discrete distribution). It has three parameters:
# - n: number of trials
# - p: probability of occurence of each trial
# - size: the shape of the returned array

x = random.binomial(n = 15, p= 0.3, size=1000)
#sns.displot(x)
#plt.show()

# Poisson distribution. It has two parameters:
# - lam: number of occurrences
# - size: the shape of the returned array

x = random.poisson(lam = 2, size = 1000)
#sns.displot(x)
#plt.show()

# and so on.. There are also methods for:
# uniform, logistic, multinomial, exponential, chi square, rayleigh, pareto and zipf distributions
# ------------------ RANDOM -----------------------


# ------------------ UFUNC -----------------------

# ufunc = universal functions. They are numpy functions that operate on the ndarray object
# They are used to implement vectorization in numpy which is way faster than iterating over elements.
# They also provide broadcasting and additional methods like reduce, accumulate etc. that are very helpful for computation.
# ufuncs take additional arguments, like:
# - where: boolean array or condition defining where the operations should take place
# - dtype: definig the return type of elements
# - out: output array where the return value should be copied

# Vectorization consists into convert iterative statements into a vector based operation.

# example without the ufunc:

x = [1,2,3,4]
y = [5,6,7,8]
z = []
for i,j in zip(x,y):    # zip returns a collection of tuples pairing the elements of the same indez. ehre it would be [(1,6), (2,7), (3,8), (4,9),(5,10)]
    z.append(i+j)
print(z)

# using the ufunc:
z = np.add(x,y)
print(z)


# personalized ufunctions can be added to the numpy ufunc library using frompyfunc(). This method takes in three arguments:
# - function: the name of the function
# - inputs: the number of input arguments (arrays)
# - outputs: the number of output arrays

def myAdd(x,y):
    return x + y
myAdd = np.frompyfunc(myAdd,2,1)
print(myAdd([1,2,3,4], [5,6,7,8])) 

# to check if a function is an ufunc or not --> print(type(np.name_function))


# basic arithmetic

# add
arr1 = np.array([10,11,12,13,14,15])
arr2 = np.array([20,21,22,23,24,25])
newArr = np.add(arr1,arr2)
print(newArr)

#subtract
arr1 = np.array([10,11,12,13,14,15])
arr2 = np.array([10,21,22,23,24,25])
newArr = np.subtract(arr1,arr2)
print(newArr)

#multiplication
arr1 = np.array([1,2,3,3,4,5])
arr2 = np.array([6,7,8,9,10,11])
newArr = np.multiply(arr1,arr2)
print(newArr)

#division
arr1 = np.array([10,12,14,27,33,81])
arr2 = np.array([2,6,2,3,11,9])
newArr = np.divide(arr1,arr2)
print(newArr)

#power
arr1 = np.array([2,2,2,2,2,5])
arr2 = np.array([2,3,4,5,6,5])
newArr = np.power(arr1,arr2)
print(newArr)

#absolute
arr = np.array([-1,-2,1,2,3,-4])
newArr = np.absolute(arr)
print(newArr)


# There are five ways of rounding off decimals in numpy:
# - truncation: remove the decimals, and return the float number closest to zero
# - fix: (same as truncation) remove the decimals, and return the float number closest to zero
# - rounding: it increments preceding digit or decimal by 1 if >= 5 else do nothing
# - floor: it rounds off decimal to the nearest lower integer
# - ceil: it rounds off decimal to the nearest upper integer

arr = np.trunc([-3.1666, 3.6667])   # -3, 3
print(arr)

arr = np.fix([-3.1666, 3.6667]) # -3, 3
print(arr)

arr = np.around(3.1666, 2)  # rounding to 2 decimal places
print(arr)
 
arr = np.floor([-3.1666, 3.6667])   # --> -4, 3
print(arr)

arr = np.ceil([-3.1666, 3.6667])    # --> -3, 4
print(arr)


# numpy also offers logarithms functions:
# - log2()
# - log10()
# - log() <-- natural
# - log with any base: we can use the library math using the log module. Then in order to vectorize it, we can use frompyfunc()


# summation

arr1 = np.array([1,2,3])
arr2 = np.array([1,2,3])
newArr = np.sum([arr1,arr2])    # 1+2+3+1+2+3
print(newArr)

arr1 = np.array([1,2,3])
arr2 = np.array([1,2,3])
newArr = np.sum([arr1,arr2], axis = 1)  # [1+2+3 1+2+3]
print(newArr)

# cumulative sum
arr = np.array([1,2,3])
newArr = np.cumsum(arr) # [1, 1+2, 1+2+3]
print(newArr)

# products
arr = np.array([1,2,3,4])
x = np.prod(arr)    # 1*2*3*4
print(x)

arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
x = np.prod([arr1,arr2])    # 1*2*3*4*5*6*7*8
print(x)

arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
x = np.prod([arr1,arr2], axis = 1)    # [1*2*3*4 5*6*7*8]
print(x)

# cumulative product
arr1 = np.array([1,2,3,4])
x = np.cumprod(arr1)    # [1, 1*2, 1*2*3, 1*2*3*4]
print(x)


# discrete difference: it means subtracting two successive elements
arr = np.array([10,15,25,5])
newArr = np.diff(arr)   # [15-10, 25-15, 5-25]
print(newArr)

# we can also provides the number of steps the operation needs to do
arr = np.array([10,15,25,5])
newArr = np.diff(arr,2)   # [15-10, 25-15, 5-25] --> result of 1st step: [5,10,-20] --> [10-5, -20-10] --> result of 2nd step: [5,-30]
print(newArr)


# lowest common multiple:
num1 = 4
num2 = 6
x = np.lcm(num1,num2)
print(x)

# lowest common multiple in arrays
arr = np.array([3,6,9])
x = np.lcm.reduce(arr)  # reduce works as always, taking two elements, it performs a specifi action... then it takes the result, the third element and perform the action, and so on...
print(x)


# same thing for greates common divisor. The ufunc is called gcd

# numpy provides methods for trigonometric and hyperbolic functions

# it also provides set operations: unique, union, intersection, difference, etc.

# ------------------ UFUNC -----------------------


# ------------------ BROADCASTING -----------------

# Very powerful.
# Given an element-wise operation among two arrays A and B of shape sA and sB.
# 1. The arrays must have all the same length of shape, therefore we can force 1 so that the have the same number of axis
# 2. For every axis k (from the last to the first), dimensions are compatible if and only if:
#       -   sA[k] == sB[k] || sA[k] == 1 || sA[k] == 1
# 3. If this control is passed, then the operation can be done and the result will have the shape: R[k] = max(sA[k], sB[k]) for every k

arr1 = np.array([1,2,3]).reshape(3,1)   # column vector
arr2 = np.array([5,5,5,5,5,5]).reshape(3,2) # matrix 3x2
newArr = arr1+arr2  # here what happens is that the columnn vector is taken and it is added to every column of the matrix
print(newArr)