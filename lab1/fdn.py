#! python3

import matplotlib.pyplot as plt
import numpy as np
import random as rnd

def back_diff(u_arr, p):
    if len(u_arr) < 2:
        print("ERRRRORRRR: U_arr has insufficient size!!!")
        exit(1)
    if p == 1:
        return u_arr[-1] - u_arr[-2]
    return back_diff(u_arr, p - 1) - back_diff(u_arr[:-1], p - 1)

def num_diff_array(func, x, h, step) -> float:
    return (func(x + 3 * h) - func(x - 3 * h)) / 6.0 / step

def J(U, func):
    size = len(U)
    res = np.zeros((size, size))
    step = 1e-5
    h = np.zeros(size)

    for i in range(size):
        h[i] = step
        res[i] = num_diff_array(func, U, h, step)
        h[i] = 0

    return res

def f(U):
    return np.array([U[1], U[0] * U[0] - 1])

def next_u_1(U, tau):
    return np.copy(U) + tau * (np.array(f(U)))

def next_u_2(U, tau, u_arr):
    return np.copy(U) + tau * (3.0/2.0 * f(u_arr[-1]) - 1.0/2.0 * f(u_arr[-2]))

def next_u_3(U, tau, u_arr):
    return np.copy(U) + tau * (23.0/12.0 * f(U) - 16.0/12.0 * f(u_arr[-1]) + 5.0/12.0 * f(u_arr[-2]))

def fdn_next_u(alpha_arr, beta, tau, func, u_arr):
    # u_new = tau * beta * func(u_new) + sum(alpha[i] * u_old[i])
    # u_new = gamma * func(u_new) + delta
    u_last = u_arr[-1]
    delta = (alpha_arr[2] * u_last + alpha_arr[1] * u_arr[-2] + alpha_arr[0] * u_arr[-3])
    gamma = tau * beta
    u_new = gamma * f(u_last) + delta

    while (np.linalg.norm(u_last - u_new) > 1e-4):
        u_last = u_new
        u_new = gamma * f(u_last) + delta

    return u_new

def lor_system(t, y):
    x, y, = y
    dxdt = y
    dydt = x * x - 1
    return np.array([dxdt, dydt])

def main():
    np.set_printoptions(floatmode='maxprec', suppress=True)
    tau = 1e-3
    rnd_size = 0.01

    plot_radius = 100
    u0 = [-1.0, 0.0]

    alpha_arr = [2.0/11.0, -9.0/11.0, 18.0/11.0]
    beta = 6.0/11.0

    for point in range(600):
        u = [u0[0] + rnd_size * rnd.randrange(-5, 500), u0[1] + rnd_size * rnd.randrange(-200, 200)]
        u_arr = [u]

        last_u = u
        res = next_u_1(last_u, tau)
        u_arr.append(res)

        last_u = np.copy(res)
        res = next_u_2(last_u, tau, u_arr)
        u_arr.append(res)

        for jojo in range(10):
            last_u = np.copy(res)
            res = next_u_3(last_u, tau, u_arr)
            u_arr.append(res)

        for t in range(900):
            res = fdn_next_u(alpha_arr, beta, tau, f, u_arr)
            if (res[0] * res[0] + res[1] * res[1]) > plot_radius:
                break
            u_arr.append(res)

        # if (res[0] * res[0] + res[1] * res[1]) > plot_radius * plot_radius:
        #     break

        x_arr, y_arr = zip(*u_arr)
        print(f"\rPlotting point = {point}", end="")
        plt.plot(x_arr, y_arr, '.-', ms=1)
    plt.grid()

    print("\nDone")

    plt.title("Phase trajectory")
    plt.ylabel("y")
    plt.xlabel("x")

    plt.show()

if __name__ == "__main__":
    main()
