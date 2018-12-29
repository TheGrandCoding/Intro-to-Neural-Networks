from matplotlib import pyplot as plt
from random import shuffle
import numpy as np

#project: predicting flower types
#watch https://youtu.be/ZzWaow1Rvho for context

#in the form [[length, width, type (0 is a blue flower, 1 is red)]]
data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, 0.5, 1],
        [2,   0.5, 0],
        [5.5, 1,   1],
        [1,   1,   0]]

mysteryFlower = [4.5, 1]

#the network

#       o    flower type
#      / \     w1, w2, b (weights and biases)
#     o   o  length, width

w1 = np.random.randn() #selects a random float from the 'normal' (?)
                       #very basically between -3 and 3
w2 = np.random.randn()
b = np.random.randn()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x): #this is the 'derivative of sigmoid'
    return sigmoid(x) * (1 - sigmoid(x))

#----------training loop-----------#
learningRate = 2 #read on to see why this matters
                 #try changing this!!
costs = []
for i in range(100000):
    ri = np.random.randint(len(data)) #ri = random index
                                      #used to pick a random data in the list
    flower = data[ri]

    z = (flower[0] * w1) + (flower[1] * w2) + b #the weighted average of the point's features

    pred = sigmoid(z)

    target = flower[2] #the type of flower
    cost = (pred - target) ** 2 #how good the prediction is (the lower the cost, the better

    costs.append(cost)
    #here's the funky derivative part
    #(i think) derivatives just show how something changes when you change something linked to it
    #apparently there's an automatic way to find derivatives
    #but we're putting ourselves through the pain of doing it ourselves
    #for 'demonstration purposes'

    #breaking down var names: dVar_Anothervar
    #                         d = derivative of
    #                         _ = with respect to
    #except sometimes there's another d behind the second var
    #dont know what this means

    dcost_pred = 2 * (pred - target)

    dpred_dz = sigmoid_prime(z)

    dz_dw1 = flower[0] #the derivative of an operation with respect to one of its operands is always the other operand
                       #i.e w1 x length = z
                       #d of z _ w1 = length
    dz_dw2 = flower[1]
    dz_db = 1

    #derivative part almost over
    #now, using the chain rule, we can get the d of cost _ w1, _w2 and _b.

    dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
    dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
    dcost_db = dcost_pred * dpred_dz * dz_db

    #derivative part over!!
    #now we tweak w1, w2 and b according to those derivative values
    #but only by a fraction of those values (learningRate) so we don't over-shoot

    w1 -= dcost_dw1 * learningRate
    w2 -= dcost_dw2 * learningRate
    b -= dcost_db * learningRate

#----------------------------------

#---------predictions!---------------
def predict(m1, m2):
    pred = sigmoid((m1 * w1) + (m2 * w2) + b)
    print("I think this flower is", "red" if pred > 0.5 else "blue")
    print("I am", (1 - pred) * 100 if pred < 0.5 else pred * 100 , "% sure")

shuffle(data)
for f in data:
    #f is one flower
    predict(f[0], f[1])
    print("(It was actually", "red)" if f[2] == 1 else "blue)" )

    print("")

print("Finally, the mystery flower...")
predict(mysteryFlower[0], mysteryFlower[1])


#--------visualising-----------#

#-----stuff about sigmoid
##X = np.linspace(-20, 20, 100) #generates numbers with regular intervals
##                           #10 numbers from -5 to 5
##Y = sigmoid(X)
##
##plt.plot(X, Y, c='r')
##plt.plot(X, sigmoid_prime(X), c='b') #this shows the *rate* that the graph above
##                                     #is changing
##plt.show()

#-----scatter data
##plt.grid()
##plt.axis([0, 6, 0, 6])
##for i in range(len(data)):
##    point = data[i]
##    plt.scatter(point[0], point[1], c="r" if point[2] == 1 else "b")
##
##plt.show()

#-----cost data
plt.plot(costs)
plt.title("How the cost (error) of the AI decreases over every iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
