from neural_network import NeuralNetwork
from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR

print "Testing AND gate implementation.\nAND Truth Table:"
_and = AND()
print "INPUT_1\tINPUT_2\tAND_RESULT"
print "True\tTrue\t"+str(_and(True, True))
print "True\tFalse\t"+str(_and(True, False))
print "False\tTrue\t"+str(_and(False, True))
print "False\tFalse\t"+str(_and(False, False))

print "\nTesting OR gate implementation.\nOR Truth Table:"
_or = OR()
print "INPUT_1\tINPUT_2\tOR_RESULT"
print "True\tTrue\t"+str(_or(True, True))
print "True\tFalse\t"+str(_or(True, False))
print "False\tTrue\t"+str(_or(False, True))
print "False\tFalse\t"+str(_or(False, False))

print "\nTesting NOT gate implementation.\nNOT Truth Table:"
_not = NOT()
print "INPUT_1\tNOT_RESULT"
print "True\t"+str(_not(True))
print "False\t"+str(_not(False))

print "\nTesting XOR gate implementation.\nXOR Truth Table:"
_xor = XOR()
print "INPUT_1\tINPUT_2\tXOR_RESULT"
print "True\tTrue\t"+str(_xor(True, True))
print "True\tFalse\t"+str(_xor(True, False))
print "False\tTrue\t"+str(_xor(False, True))
print "False\tFalse\t"+str(_xor(False, False))
