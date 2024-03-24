#! python3

import matplotlib.pyplot as plt
import numpy as np

def f(t, U):
    a = 1e3
    A = 0.4
    omega = 10.0
    return [a * (-(U[0]**3/3 - U[0]) + U[1]), -U[0] + A * np.cos(omega * t)]

def k(t, U, s, tau, beta_matr, alpha_arr):
    if s <= 1:
        return f(U)

    k_arr = np.empty((s + 1, 2))
    k_arr[0] = f(t, U)

    for i in range(1, s):
        U_new = np.copy(U)
        for j in range(i):
            U_new += tau * beta_matr[i - 1][j] * k_arr[j]
        k_arr[i] = f(t + tau * alpha_arr[i], U_new)

    return k_arr

def next_u(U, tau, gamma_arr, k_arr):
    res = [0, 0]

    for i in range(len(gamma_arr)):
        res += k_arr[i] * gamma_arr[i] * tau

    return np.copy(U) + res

def main():
    np.set_printoptions(floatmode='maxprec', suppress=True)

    tau = 1e-4

    u0 = [2.0, 0.0]

    beta_matr = np.zeros((2, 2))
    beta_matr[0][0] = 0.25
    beta_matr[0][1] = 0.25 - np.sqrt(3)/6
    beta_matr[1][0] = 0.25 + np.sqrt(3)/6
    beta_matr[1][1] = 0.25

    alpha_arr = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
    gamma_arr = np.array([0.5, 0.5])

    u_arr = [u0]
    res = u0

    for t in np.arange(0.0, 200.0, tau):
        print(f"\rCalculating t = {t}", end="")
        last_u = res
        k_arr = k(t, last_u, 2, tau, beta_matr, alpha_arr)
        res = next_u(last_u, tau, gamma_arr, k_arr)
        u_arr.append(res)
    print()

    print("Plotting...")
    x_arr, y_arr = zip(*u_arr)
    plt.plot(x_arr, y_arr, '.-', ms=1)
    plt.grid()

    print("\nDone")

    plt.title("Phase trajectory")
    plt.ylabel("y")
    plt.xlabel("x")

    plt.savefig("test.png")

if __name__ == "__main__":
    main()
