#!/usr/bin/env python3
# File       : newton_salad.py
# Description: Newton's method using Salad AD
import numpy as np
from salad import *


# f = lambda x: x + 2*x + 3/x
# J = lambda x, eps: (f(x + eps) - f(x)) / eps # Finite-Difference approximation of J

def newton(x_k, tol=1.0e-8, max_it=100, eps=1.0e-8):
    root = None
    for k in range(max_it):

        x = Variable(x_k, label='x_k')
        f = 2*x**3  - 2*x - 5

        dx_k = -1 * f.val / f.der['x_k']
        print(dx_k)
        # dx_k = -f(x_k) / J(x_k, eps)

        if abs(dx_k) < tol:
            root = x_k + dx_k
            # print(f"Found root {root:e} at iteration {k+1}")
            print("Found root ", root, " at iteration ", k+1)
            # print(f(root)) # Function cannot currently be called like this
            break
        # print(f"Iteration {k+1}: Delta x = {dx_k:e}")
        print("Iteration ", k+1, ": Delta x = ", dx_k)
        x_k += dx_k
    return root



if __name__ == "__main__":
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description="Newton-Raphson Method")
        parser.add_argument('-g', '--initial_guess', type=float, help="Initial guess", required=True)
        parser.add_argument('-t', '--tolerance', type=float, default=1.0e-8, help="Convergence tolerance")
        parser.add_argument('-i', '--maximum_iterations', type=int, default=100, help="Maximum iterations")
        return parser.parse_args()

    args = parse_args()
    # newton(f, J, args.initial_guess, args.tolerance, args.maximum_iterations)
    newton(x_k=args.initial_guess, tol=args.tolerance, max_it=args.maximum_iterations)


