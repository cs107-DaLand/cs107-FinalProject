#!/usr/bin/env python3
# File       : newton_salad.py
# Description: Newton's method using Salad AD. Only works for single variable case

"""
To use right now, you can do 
>>> from cs107_salad.Optimization.newton_draft import *
>>> f = '2*x**3  - 2*x - 5'
>>> newton(f, 2, 1.0e-8, 100)
-0.3181818181818182
Iteration  1 : Delta x =  -0.3181818181818182
-0.07684490502597054
Iteration  2 : Delta x =  -0.07684490502597054
-0.004361047546956799
Iteration  3 : Delta x =  -0.004361047546956799
-1.3684177142438221e-05
Iteration  4 : Delta x =  -1.3684177142438221e-05
-1.344910547632496e-10
Found root  1.6005985449336209  at iteration  5
2
1.6005985449336209
"""

import numpy as np
from ..Forward import salad
from ..Forward import utils


def newton(f, x_k, tol=1.0e-8, max_it=100, eps=1.0e-8):
    root = None
    for k in range(max_it):

        x = {'x': x_k}
        forward = salad.Forward(x, [f])

        # dx_k = -f(x_k) / J(x_k, eps)
        dx_k = -1 * forward.results[0].val / forward.results[0].der['x']
        print(dx_k)

        if abs(dx_k) < tol:
            root = x_k + dx_k
            print("Found root ", root, " at iteration ", k+1)
            print(eval(f[0], {'x': root}))
            break
        print("Iteration ", k+1, ": Delta x = ", dx_k)
        x_k += dx_k
    return root


def main():
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description="Newton-Raphson Method")
        parser.add_argument('-g', '--initial_guess', type=float, help="Initial guess", required=True)
        parser.add_argument('-t', '--tolerance', type=float, default=1.0e-8, help="Convergence tolerance")
        parser.add_argument('-i', '--maximum_iterations', type=int, default=100, help="Maximum iterations")
        return parser.parse_args()

    args = parse_args()

    f = '2*x**3  - 2*x - 5'
    newton(f=f, x_k=args.initial_guess, tol=args.tolerance, max_it=args.maximum_iterations)

if __name__ == "__main__":
    main()


