#! python3

import matplotlib.pyplot as plt
import numpy as np
import random as rnd

def f(U):
    return np.array([U[1], U[0] * U[0] - 1])

def next_u_1(U, tau):
    return np.copy(U) + tau * (np.array(f(U)))

def next_u_2(U, tau, u_arr):
    return np.copy(U) + 2 * tau * f(U) + np.array(u_arr[-2])

def next_u_3(U, tau, u_arr):
    return np.copy(U) + tau * f(U) - 3.0/2.0 * np.array(U) + 3.0 * np.array(u_arr[-2]) - 0.5 * np.array(u_arr[-3])

def main():
    np.set_printoptions(floatmode='maxprec', suppress=True)

    tau = 1e-5
    rnd_size = 0.001

    plot_radius = 100
    u0 = [1.0, 0.0]

    u_arr = [u0]

    for point in range(800):
        u = [u0[0] + rnd_size * rnd.randrange(-5, 5), u0[1] + rnd_size * rnd.randrange(-5, 5)]
        x_arr = np.array([u[0]])
        y_arr = np.array([u[1]])

        last_u = u
        res = next_u_1(last_u, tau)
        x_arr = np.append(x_arr, res[0])
        y_arr = np.append(y_arr, res[1])
        u_arr.append(res)

        last_u = np.copy(res)
        res = next_u_2(last_u, tau, u_arr)
        x_arr = np.append(x_arr, res[0])
        y_arr = np.append(y_arr, res[1])
        u_arr.append(res)

        for t in range(600):
            last_u = np.copy(res)
            res = next_u_3(last_u, tau, u_arr)
            if (res[0] * res[0] + res[1] * res[1]) > plot_radius:
                break
            x_arr = np.append(x_arr, res[0])
            y_arr = np.append(y_arr, res[1])
            u_arr.append(res)

        print(f"\rPlotting point = {point}", end="")
        plt.plot(x_arr, y_arr, '.-', ms=1)
    plt.grid()

    print("\nDone")

    plt.title("Phase trajectory")
    plt.ylabel("y")
    plt.xlabel("x")

    plt.show()
    # plt.savefig("jojo.png")

if __name__ == "__main__":
    main()
