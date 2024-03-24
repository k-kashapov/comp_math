#! python3

import matplotlib.pyplot as plt
import numpy as np

# Constants
a = 1e4
A = 1.1
omega = 10000.0

def f(t, U):
    if U[0] > 1000:
        print("\n", t, U)
    return [a * (-(U[0]**3/3 - U[0]) + U[1]), -U[0] + A * np.cos(omega * t)]

def next_u_1(t, u_arr, tau, func):
    u_last = u_arr[-1]
    eps = 1e-3

    u_next = u_last + tau * np.array(func(t, u_last))
    # print('u_next_1 =', u_next)

    while np.linalg.norm(u_next - u_last) > eps:
        u_last = u_next
        u_next = u_last + tau * np.array(func(t, u_last))
        # print('u_next_1 =', u_next)

    return u_next

def next_u_2(t, u_arr, tau, func):
    u_last = u_arr[-1]
    eps = 1e-3

    u_next = 2.0/3.0 * (tau * np.array(func(t, u_last)) + 2.0 * np.array(u_arr[-1]) - 0.5 * np.array(u_arr[-2]))

    # print('u_next_2 =', u_next)
    while np.linalg.norm(u_next - u_last) > eps:
        u_last = u_next
        u_next = 2.0/3.0 * (tau * np.array(func(t, u_last)) + 2.0 * np.array(u_arr[-1]) - 0.5 * np.array(u_arr[-2]))
        # print('u_next_2 =', u_next)
    return u_next

def next_u_3(t, u_arr, tau, func):
    u_last = u_arr[-1]
    eps = 1e-3

    u_next = 6.0/11.0 * (tau * np.array(func(t, u_last)) + 3.0 * np.array(u_arr[-1]) - 1.5 * np.array(u_arr[-2]) + 1.0/3.0 * np.array(u_arr[-3]))
    # print('u_next_3 =', u_next)

    while np.linalg.norm(u_next - u_last) > eps:
        u_last = u_next
        u_next = 6.0/11.0 * (tau * np.array(func(t, u_last)) + 3.0 * np.array(u_arr[-1]) - 1.5 * np.array(u_arr[-2]) + 1.0/3.0 * np.array(u_arr[-3]))
        # print('u_next_3 =', u_next)

    return u_next

def main():
    np.set_printoptions(floatmode='maxprec', suppress=True)

    tau = 1e-5
    #      x    y
    u0 = [2.0, 0.0]

    u_arr = [u0]
    res = u0

    last_u = res
    res = next_u_1(0.0, u_arr, tau, f)
    u_arr.append(res)

    last_u = res
    res = next_u_2(tau, u_arr, tau, f)
    u_arr.append(res)

    # exit(1)

    for t in np.arange(tau * 2, 200.0, tau):
        print(f"\rCalculating t = {t}", end="")
        last_u = res
        res = next_u_3(t, u_arr, tau, f)
        u_arr.append(res)
    print()

    print("Plotting...")
    x_arr, y_arr = zip(*u_arr)
    plt.plot(x_arr[::100], y_arr[::100], '.-', ms=1)
    plt.grid()

    print("\nDone")

    plt.title(f"Phase trajectory, A = {A}, a = {a}, omega = {omega}")
    plt.ylabel("y")
    plt.xlabel("x")

    plt.show()

if __name__ == "__main__":
    main()
