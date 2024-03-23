#! /bin/python3.10

# Номер IV.12.5 (г)

import math
import matplotlib.pyplot as plt
import numpy as np

def mpi_x(x: float, y: float) -> float:
    return 0.5 - math.cos(y - 2)

def mpi_y(x: float, y: float) -> float:
    return math.sin(x + 2) - 1.5

def J(x: float, y:float):
    ret = np.zeros((2, 2))
    ret[0][0] = math.cos(x + 2)
    ret[0][1] = -1
    ret[1][0] = 1
    ret[1][1] = -math.sin(y - 2)
    return ret

def F(x: float, y: float):
    return [math.sin(x + 2) - y - 1.5, x + math.cos(y - 2) - 0.5]

def main():
    eps: float = 1e-3

    # ===========< MPI >============
    print("MPI:")

    x_last = 10.0
    y_last = 10.0

    xs = [x_last]
    ys = [y_last]

    x_new = mpi_x(x_last, y_last)
    y_new = mpi_y(x_last, y_last)
    while (abs(x_new - x_last) > eps) and (abs(y_new - y_last) > eps):
        print(x_new, y_new)
        x_last = x_new
        y_last = y_new
        xs.append(x_last)
        ys.append(y_last)
        x_new = mpi_x(x_last, y_last)
        y_new = mpi_y(x_last, y_last)

    plt.grid()

    plt.title("X vs Y. MPI")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.plot(xs, ys, '.-', ms=8.0)
    plt.savefig("img/system_mpi.png")
    plt.clf()

    # ===========< Newton >============
    print("Newton method:")
    x_last = 5.0
    y_last = 5.0

    xs = [x_last]
    ys = [y_last]

    print(x_last, y_last)
    dF = J(x_last, y_last)
    x_new = x_last - (np.linalg.inv(dF) @ F(x_last, y_last))[0]
    y_new = y_last - (np.linalg.inv(dF) @ F(x_last, y_last))[1]
    
    xs.append(x_new)
    ys.append(y_new)

    while (abs(x_new - x_last) > eps) and (abs(y_new - y_last) > eps):
        x_last = x_new
        y_last = y_new
        print(x_last, y_last)

        dF = J(x_last, y_last)
        print(dF)
        print(np.linalg.inv(dF))
        x_new = x_last - (np.linalg.inv(dF) @ F(x_last, y_last))[0]
        y_new = y_last - (np.linalg.inv(dF) @ F(x_last, y_last))[1]
        
        xs.append(x_new)
        ys.append(y_new)
        

    plt.grid()

    plt.title("X vs Y. Newton")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.plot(xs, ys, '.-', ms=5.0)
    plt.savefig("img/system_newton.png")
    plt.clf()

if __name__ == '__main__':
    main()
