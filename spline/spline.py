#! python3

import matplotlib.pyplot as plt
import numpy as np

def gauss_direct(matrix, f):
    rows, cols = matrix.shape
    for col in range(cols):
        max_idx: int = np.argmax(abs(matrix[col:rows, col]))
        max_idx += col

        if col != max_idx:
            matrix[[col, max_idx]] = matrix[[max_idx, col]]
            f[col], f[max_idx] = f[max_idx], f[col]

        f[col]      /= matrix[col][col]
        matrix[col] /= matrix[col][col]

        for row in range(col + 1, rows):
            if matrix[row][col] != 0:
                mult = matrix[row][col]
                f[row] -= f[col] * mult
                matrix[row] -= matrix[col] * mult

    return matrix

def gauss_inverse(matrix, f):
    rows, cols = matrix.shape
    for col in range(cols - 1, -1, -1):
        max_idx: int = np.argmax(abs(matrix[:, col]))
        max_idx += col

        f[col]      /= matrix[col][col]
        matrix[col] /= matrix[col][col]

        for row in range(col - 1, -1, -1):
            if matrix[row][col] != 0:
                mult = matrix[row][col]
                f[row] -= f[col] * mult
                matrix[row] -= matrix[col] * mult

    return matrix

def GaussSolution(matrix, f):
    x = np.copy(f)
    gauss_res = gauss_direct(np.copy(matrix), x)

    # print("\nAfter direct gauss:")
    # print(gauss_res)
    # print(x)

    # print("\nAfter inverse:")
    gauss_res = gauss_inverse(gauss_res, x)
    # print(gauss_res)
    print("\nSolution:\n", x)
    return x

def div_diff(xs, ys, k_start, k_end):
    if k_start == k_end:
        return ys[k_start]
    return (div_diff(xs, ys, k_start + 1, k_end) - div_diff(xs, ys, k_start, k_end-1)) / (xs[k_end] - xs[k_start])

# Warning: i >= 1
def u(xs, ys, i):
    return (div_diff(xs, ys, i, i + 1) - div_diff(xs, ys, i - 1, i)) / (xs[i + 1] - xs[i - 1])

def coefs(c, a, i, h, xs, ys):
    coefs = np.zeros(4)
    coefs[0] = a[i]
    coefs[1] = c[i] * h / 3 + c[i - 1] / 6 * h + div_diff(xs, ys, i - 1, i)
    coefs[2] = c[i]
    coefs[3] = (c[i] - c[i - 1]) / h
    return coefs

def Spline(xs, ys, x_tgt, c, h):
    i = len(ys) - 1
    xi = xs[i]

    for j in range(len(xs)):
        if x_tgt <= xs[j]:
            xi = xs[j]
            i = j
            break

    coeffs = coefs(c, ys, i, h, xs, ys)

    return coeffs[0] + coeffs[1] * (x_tgt - xi) + (coeffs[2] / 2) * (x_tgt - xi)**2 + (coeffs[3] / 6) * (x_tgt - xi)**3

def main():
    np.set_printoptions(floatmode='maxprec', precision=3, suppress=True)
    years      = [1910., 1920., 1930., 1940., 1950., 1960., 1970., 1980., 1990., 2000.]
    population = [92228496., 106021537., 123202624., 132164569., 151325798., 179323175., 203211926., 226545805., 248709873., 281421906.]

    n = len(years) - 2
    cs = np.zeros((n, n))
    h = years[1] - years[0]
    rs = np.zeros(n)

    for i in range(0, n):
        rs[i] = 6 * u(years, population, i)
        if i < n - 1:
            cs[i + 1][i] = 0.5
        cs[i][i] = 2
        if i > 0:
            cs[i - 1][i] = 0.5

    c = GaussSolution(cs, rs)

    c = np.append(c, 0)
    c = np.insert(c, 0, 0)
    print(c)

    x_plt = np.arange(1910, 2020, 1)
    y_plt = []
    for x in x_plt:
        y_plt.append(Spline(years, population, x, c, h))

    plt.plot(x_plt, y_plt, '.-')
    plt.scatter(years, population)
    plt.grid()
    
    # plt.yscale('log')
    plt.title("Values, Spline")
    plt.ylabel("Population")
    plt.xlabel("Year")

    # plt.show()
    plt.savefig("img/SplineBest.png")

if __name__ == "__main__":
    main()
