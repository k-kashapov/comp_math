#! python3

import matplotlib.pyplot as plt
import numpy as np
import random as rnd

def f(U):
    return [U[1], U[0] * U[0] - 1]

def k(U, s, tau, A):
    if s <= 1:
        return f(U)

    k_arr = np.empty((s + 1, 2))
    k_arr[0] = f(U)

    for i in range(1, s + 1):
        U_new = np.copy(U)
        for j in range(i):
            U_new += tau * A[i - 1][j] * k_arr[j]
        k_arr[i] = f(U_new)

    return k_arr

def next_u(U, tau, b, k_arr):
    res = [0, 0]

    for i in range(len(b)):
        res += k_arr[i] * b[i] * tau

    return np.copy(U) + res

def main():
    np.set_printoptions(floatmode='maxprec', suppress=True)

    tau = 1e-2
    rnd_size = 0.001

    plot_radius = 100
    u0 = [-1.0, 0.0]

    A = np.zeros((3, 3))
    A[0][0] = 0.5
    A[1][1] = 0.5
    A[2][2] = 1.0

    b = np.array([1.0/6, 1.0/3, 1.0/3, 1.0/6])

    for point in range(800):
        u = [u0[0] + rnd_size * rnd.randrange(-5, 5000), u0[1] + rnd_size * rnd.randrange(-200, 200)]
        x_arr = np.array([u[0]])
        y_arr = np.array([u[1]])

        for t in range(0, 100):
            last_u = [x_arr[-1], y_arr[-1]]
            k_arr = k(last_u, 3, tau, A)
            res = next_u(last_u, tau, b, k_arr)
            if (res[0] * res[0] + res[1] * res[1]) > plot_radius:
                break
            x_arr = np.append(x_arr, res[0])
            y_arr = np.append(y_arr, res[1])

        print(f"\rPlotting point = {point}", end="")
        plt.plot(x_arr, y_arr, '.-', ms=1)
    plt.grid()

    print("\nDone")

    # plt.xticks(np.linspace(u0[0] - rnd_size * 4, u0[0] + rnd_size * 4, 15))
    # plt.yticks(np.linspace(u0[1] - rnd_size * 4, u0[1] + rnd_size * 4, 15))

    # plt.yscale('log')
    plt.title("Phase trajectory")
    plt.ylabel("y")
    plt.xlabel("x")

    plt.show()
    # plt.savefig("jojo.png")

if __name__ == "__main__":
    main()
