#! python3

import matplotlib.pyplot as plt
import numpy as np
import random as rnd

def f(U):
    return [U[1], U[0] * U[0] - 1]

def next_u_1(U, tau, f_arr):
    f_arr.append(f(U))
    return np.copy(U) + tau * (np.array(f_arr[-1]))

def next_u_2(U, tau, f_arr):
    f_arr.append(f(U))
    return np.copy(U) + tau * (3.0/2.0 * np.array(f_arr[-1]) - 1.0/2.0 * np.array(f_arr[-2]))

def next_u_3(U, tau, f_arr):
    f_arr.append(f(U))
    return np.copy(U) + tau * (23.0/12.0 * np.array(f_arr[-1]) - 16.0/12.0 * np.array(f_arr[-2]) + 5.0/12.0 * np.array(f_arr[-3]))

def main():
    np.set_printoptions(floatmode='maxprec', suppress=True)

    tau = 1e-3
    rnd_size = 0.001

    plot_radius = 100
    u0 = [1.0, 0.0]

    f_arr = []

    for point in range(800):
        u = [u0[0] + rnd_size * rnd.randrange(-500, 500), u0[1] + rnd_size * rnd.randrange(-500, 500)]
        x_arr = np.array([u[0]])
        y_arr = np.array([u[1]])

        last_u = u
        res = next_u_1(last_u, tau, f_arr)
        x_arr = np.append(x_arr, res[0])
        y_arr = np.append(y_arr, res[1])

        last_u = np.copy(res)
        res = next_u_2(last_u, tau, f_arr)
        x_arr = np.append(x_arr, res[0])
        y_arr = np.append(y_arr, res[1])

        for t in range(600):
            last_u = np.copy(res)
            res = next_u_3(last_u, tau, f_arr)
            if (res[0] * res[0] + res[1] * res[1]) > plot_radius:
                break
            x_arr = np.append(x_arr, res[0])
            y_arr = np.append(y_arr, res[1])

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
