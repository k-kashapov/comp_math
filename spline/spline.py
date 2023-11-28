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

def coefs(c, a, i, h):
    # print(len(c), len(a), i, h)
    coefs = np.zeros(4)
    coefs[0] = a[i]
    coefs[1] = (a[i+1] - a[i]) / h - (2 * c[i] + c[i + 1]) / 3 * h
    coefs[2] = c[i]
    coefs[3] = (c[i+1] - c[i]) / 3 / h
    return coefs

def Spline(xs, ys, x_tgt, c, h):
    xi = 0
    i = 0

    for j in range(len(xs)):
        if x_tgt <= xs[j]:
            xi = xs[j]
            i = j - 1
            break

    # print(x_tgt, xs[i])

    coeffs = coefs(c, ys, i, h)

    return coeffs[0] + coeffs[1] * (x_tgt - xi) + coeffs[2] * (x_tgt - xi)**2 + (coeffs[3]) * (x_tgt - xi)**3

def main():
    np.set_printoptions(floatmode='maxprec', precision=3, suppress=True)
    years      = [1910., 1920., 1930., 1940., 1950., 1960., 1970., 1980., 1990., 2000.]
    population = [92228496., 106021537., 123202624., 132164569., 151325798., 179323175., 203211926., 226545805., 248709873., 281421906.]

    cs = np.zeros((len(years) - 2, len(years) - 2))
    h = years[1] - years[0]
    rs = np.zeros(len(years) - 2)

    for i in range(0, len(years) - 2):
        if i > 0:
            rs[i - 1] = 3/h/h * (population[i + 1] - 2*population[i - 1] + population[i - 1])
            cs[i][i - 1] = 1
        cs[i][  i  ] = 4
        if i < len(years) - 3:
            cs[i][i + 1] = 1

    print(cs)

    c = GaussSolution(cs, rs)
    c = np.insert(c, 0, 0)
    c = np.append(c, 0)

    print(c)

    x_plt = np.arange(1910, 2000, 1)
    y_plt = []
    for x in x_plt:
        # print(x, end="\r")
        y_plt.append(Spline(years, population, x, c, h))

    plt.plot(x_plt, y_plt, '.-')
    plt.scatter(years, population)
    plt.grid()
    
    plt.title("Values, Spline")
    plt.ylabel("Population")
    plt.xlabel("Year")

    # plt.show()
    plt.savefig("img/SplineValue.png")

if __name__ == "__main__":
    main()
