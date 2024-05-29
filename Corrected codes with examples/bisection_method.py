
import numpy as np
import sympy as sp
from sympy import *
from colors import bcolors
from sympy.utilities.lambdify import lambdify

def max_steps(a, b, err):
    s = int(np.floor(- np.log2(err / (b - a)) / np.log2(2) - 1))
    return s

def find_derivative(expression):
    x = sp.symbols('x')
    derivative = sp.diff(expression, x)
    print(derivative)
    return derivative

def bisection_method(f, a, b, tol):
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("The scalars a and b do not bound a root")

    c, k = 0, 0
    steps = max_steps(a, b, tol)
    #print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))
    while abs(b - a) > tol and k < steps:
        c = a + (b - a) / 2
        if f(c) == 0:
            return c
        if f(a) == 0:
            return a
        if f(b) == 0:
            return b
        if np.sign(f(a)) == np.sign(f(c)):
            a = c
        else:
            b = c
        k += 1
        #print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(k, a, b, f(a), f(b), c, f(c)))
    if f(c) > 0.001:
        raise Exception("The scalars a and b do not bound a root")
    return c

def find_all_roots(f, interval, tol):
    f1 = lambdify(x, f)
    a, b = interval
    roots = []
    interval1 = (b - a)/10
    while a <= b:
        flag = -1
        try:
            root = bisection_method(f1, a, a+interval1, tol)
            if root < a or root > b:  # Check if the root is outside the current interval
                continue

            if len(roots) == 0:
                roots.append(round(root, 5))
            else:
                for i in roots:
                    if i == root:
                        flag = 0
                        break
                if flag == -1:
                    roots.append(round(root, 5))
        except Exception as e:
            pass

        a += interval1
    return roots

if __name__ == '__main__':
    tol = 1e-6
    x = sp.symbols('x')


    f = x**3 - 4*sin(x)

    fTAG = sp.diff(f)
    interval = (-5, 5)

    roots = find_all_roots(f, interval, tol)
    Extreme_Points = find_all_roots(fTAG, interval, tol)

    f = lambdify(x, f)
    for i in Extreme_Points:
        if 0+tol >= f(i) >= 0-tol:
            roots.append(round(i, 5))
    print(bcolors.OKBLUE, f"\nThe equation f(x) has an approximate root at x = {roots}", bcolors.ENDC, )