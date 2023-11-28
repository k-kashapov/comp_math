#! python3

import matplotlib.pyplot as plt
import numpy as np
import math

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

def func(x, coefs):
    x_n = 1
    ret = 0
    for i in range(len(coefs) - 1, -1, -1):
        ret += coefs[i] * x_n
        x_n *= x

    return ret

def func2(x, coefs):
    ret = 0
    for i in range(len(coefs) - 1, -1, -1):
        ret += math.exp(coefs[i] * x)

    return ret

# approx with P_power(x)
def main():
    np.set_printoptions(floatmode='maxprec', suppress=True)
    years      = [1910., 1920., 1930., 1940., 1950., 1960., 1970., 1980., 1990., 2000.]
    population = [92228496., 106021537., 123202624., 132164569., 151325798., 179323175., 203211926., 226545805., 248709873., 281421906.]

    power = 3

    n = len(years)
    
    A = np.zeros((power + 1, power + 1))

    sums = np.zeros(2 * power + 1)
    for i in range(2 * power + 1):
        for year in years:
            sums[i] += pow(year, i)

    ysums = np.zeros(power + 1)
    for i in range(power + 1):
        for j in range(n):
            ysums[i] += population[j] * pow(years[j], i)

    for x in range(power + 1):
        for y in range(power + 1):
            A[x][y] = sums[4 - (x + y)]

    res = GaussSolution(A, ysums)

    x_check = np.arange(years[0], years[-1] + 10)
    predictions = []

    for x in x_check:
        predictions.append(func(x, res))

    for i in range(n):
        predict = func(years[i], res)
        print(f"year: {years[i]}, actual: {population[i]}, predicted: {predict}, div: {predict/population[i]}")

    plt.scatter(years, population)
    plt.plot(x_check, predictions, '.-')
    plt.grid()
    
    plt.title(f"Values, LMS method, power={power}")
    plt.ylabel("Population")
    plt.xlabel("Year")

    # plt.show()
    plt.savefig(f"img/Square{power}.png")

if __name__ == "__main__":
    main()
