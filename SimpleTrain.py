from math import sqrt
def cost(b): #shows how 'good' of an estimate b is,
             #depending on how close b is to the actual value
             #(we just made up a random value, 4)
             #the lower the result, the better it is
    return (b-4) ** 2

def num_slope(b): #will return the gradient of the tangent to the cost(b) curve;
                  #the higher the gradient,
                  #the further it is from the actual value (4 in this case)
    h = 0.0001 #some small difference (the smaller the better)
    return((cost(b+h) - cost(b)) / h)# just dy/dx

def slope(b):
    return 2 * (b-4) #should output the same as num_slope,
                     #only the methode to get there is simpler
                     #(using the derivative)

def compareSlopes(b):
    print("num_slope ", num_slope(b))
    print("slope ", slope(b))

b = 5

def update(): #this is how using slopes will lead you to the correct answer
    global b
    b = b - (0.1 * slope(b))  #'following the tangent' a little to get closer to
                              #the actual value
    print("b: ", b)

#a training loop,
#minimising the cost over every iteration
for i in range(5000):
    update()
