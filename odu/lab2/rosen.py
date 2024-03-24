#! python3

import matplotlib.pyplot as plt
import numpy as np
import random as rnd

# Constants
a = 1e4
A = 1.2
omega = 1000.0

def f(U):
    return [a * (-(U[0]**3/3 - U[0]) + U[1]), -U[0] + A * np.cos(omega * U[2]), 1.0]

def k(u_arr, s, tau, beta_matr, alpha_arr):
    if s <= 1:
        return f(U)

    k_arr = np.empty((s + 1, 3))
    k_arr[0] = f(t, U)

    for i in range(1, s):
        U_new = np.copy(U)
        for j in range(i):
            U_new += tau * beta_matr[i - 1][j] * k_arr[j]
        k_arr[i] = f(t + tau * alpha_arr[i], U_new)

    return k_arr

def next_u(U, tau, gamma_arr, k_arr):
    res = [0, 0, 0]

    for i in range(len(gamma_arr)):
        res += k_arr[i] * gamma_arr[i] * tau

    return np.copy(U) + res

def J(U):
    ret = np.zeros((len(U), len(U)))
    ret[0][0] = -a * (U[0] * U[0] + 1)
    ret[0][1] = a
    ret[1][0] = -1
    ret[1][2] = -A * omega * np.sin(omega * U[2])
    ret[2][2] = 1
    return ret

def matrix_left(tau, func, u0):
    dF = J(u0)
    ret = np.identity(len(u0)) - (1 + 1j) / 2 * tau * dF

    return ret

def right_side(func, U, tau):
    U_new = np.copy(U)
    U_new[2] += tau / 2
    return func(U_new)

def main():
    np.set_printoptions(floatmode='maxprec', suppress=True)

    tau = 1e-4
    #      x    y    t
    u0 = [2.0, 0.0, 0.0]

    u_arr = [u0]
    res = u0

    left = matrix_left(tau, f, u0)

    for t in np.arange(0.0, 50.0, tau):
        print(f"\rCalculating t = {t}", end="")
        last_u = res
        right = right_side(f, last_u, tau)
        w = np.linalg.solve(left, right)
        res = last_u + tau * np.real(w)
        # print(res)
        u_arr.append(res)
    print()

    print("Plotting...")
    x_arr, y_arr, t_arr = zip(*u_arr)
    plt.plot(t_arr, y_arr, '.-', ms=1)
    plt.plot(t_arr, x_arr, '.-', ms=1)
    plt.grid()

    print("\nDone")

    plt.title(f"Phase trajectory, A = {A}, a = {a}, omega = {omega}")
    plt.ylabel("y")
    plt.xlabel("x")

    plt.show()

if __name__ == "__main__":
    main()
