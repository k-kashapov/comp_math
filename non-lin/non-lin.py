#! /bin/python3.10

# Номер IV.12.4 (з) 2lgx - x/2 + 1 = 0

import math
import matplotlib.pyplot as plt

def x_next(x: float) -> float:
    return 4.0 * math.log10(x) + 2.0

def deriv(func, x: float, h: float) -> float:
    return (func(x + h) - func(x)) / h

def F(x: float) -> float:
    return 2.0 * math.log10(x) - x / 2.0 + 1

def main():
    # ===========< MPI >============
    print("MPI:")
    x_last = 10.0

    xs = [x_last]

    x_new = x_next(x_last)
    while abs(x_new - x_last) > 1e-4:
        print(x_new)
        x_last = x_new
        xs.append(x_last)
        x_new = x_next(x_last)

    plt.grid()

    plt.title("X_(n+1) vs step. MPI")
    plt.ylabel("X")
    plt.xlabel("Step")

    plt.plot(xs, '.-', ms=5.0)
    plt.savefig("img/x_n_mpi.png")
    plt.clf()

    print(f"MPI final = {x_new}")

    # ===========< Newton >============
    print("Newton method:")
    x_last = 10.0
    h = 1e-3
    
    dFdx = deriv(F, x_last, h)
    x_new = x_last - F(x_last)/dFdx
    
    xs = [x_last]

    print(x_last)
    while abs(x_new - x_last) > 1e-4:
        x_last = x_new
        dFdx = deriv(F, x_last, h)
        x_new = x_last - F(x_last)/dFdx
        print(x_new)
        xs.append(x_new)

    plt.grid()

    plt.title("X_(n+1) vs step. Newton")
    plt.ylabel("X")
    plt.xlabel("Step")

    plt.plot(xs, '.-', ms=5.0)
    plt.savefig("img/x_n_newton.png")
    plt.clf()

    print(f"MPI final = {x_new}")

if __name__ == '__main__':
    main()
