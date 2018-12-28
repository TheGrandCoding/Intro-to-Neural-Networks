import random
from math import exp

def NeuralNetwork(m1, m2, w1, w2, b): #m = measurement, w = weight, b = bias
    z = (m1 * w1) + (m2 * w2) + b #the output (?)
    return sigmoid(z)

def sigmoid(x): #sigmoid 'squishes' numbers into a value between -1 and 1 (?)
    return 1 / (1 + exp(-x)) #the sigmoid function


w1 = random.uniform(-2, 2)
w2 = random.uniform(-2, 2)
b = random.uniform(-2, 2)

def go():
    m1 = float(input("length measurement"))
    m2 = float(input("width measurement"))
    print(NeuralNetwork(m1, m2, w1, w2, b))

