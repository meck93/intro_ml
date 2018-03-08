import math

# feature transformation functions
functions = [lambda x: float(x), lambda x: math.pow(x, 2), lambda x: math.exp(x), lambda x: math.cos(x), lambda x: float(1)]

print("Example of how to use lambda-functions in python for Task 1b")

for i in range(0,5):
    print("All Functions for Value:", i)
    for j in range(0,5):
        print("Function:\t", j, "\tInput\t", i, "\tValue:\t", functions[j](i))

    print()