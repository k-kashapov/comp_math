#! /bin/python3.10

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

# Номер II.10.3.Г

def gen_matrix(n: int):
    matrix  = np.zeros((n, n))
    f = np.arange(float(n), 0.0, -1.0)

    for col in range(n):
        matrix[0][col] = 1

    for row in range(1, n - 1):
        matrix[row][row - 1] = 1
        matrix[row][row]     = 10
        matrix[row][row + 1] = 1

    matrix[n - 1][n - 1] = 1
    matrix[n - 1][n - 2] = 1

    return matrix, f

def gauss_direct(matrix, f):
    rows, cols = matrix.shape
    for col in range(cols):
        max_idx: int = np.argmax(abs(matrix[col:rows, col]))
        max_idx += col

        # print(f"\nFound max at: {col}, {max_idx}\n")
        # print(matrix, "\n")

        if col != max_idx:
            matrix[[col, max_idx]] = matrix[[max_idx, col]]
            f[col], f[max_idx] = f[max_idx], f[col]
            # print("After swapping:\n", matrix)
            # print(f, "\n")

        # print(f"Before div: {f[col]}")
        # print(f"Divisor   : {matrix[col][col]}")

        f[col]      /= matrix[col][col]
        matrix[col] /= matrix[col][col]

        # print("After division:\n", matrix)
        # print(f, "\n")

        for row in range(col + 1, rows):
            if matrix[row][col] != 0:
                mult = matrix[row][col]
                # print("Before:     ", f[row])
                # print("Subtracting:", f[col] * mult)
                f[row] -= f[col] * mult
                matrix[row] -= matrix[col] * mult
                # print("After:      ", f[row])
                # print()

        # print(matrix)
        # print(f)
        # print()

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

def main():
    np.set_printoptions(floatmode='maxprec', precision=3, suppress=True)
    matrix, f = gen_matrix(100)
    x = np.copy(f)
    
    print("\nInput:")
    print(matrix)
    print(x)

    gauss_res = gauss_direct(np.copy(matrix), x)

    print("\nAfter direct gauss:")
    print(gauss_res)
    print(x)

    print("\nAfter inverse:")
    gauss_res = gauss_inverse(gauss_res, x)
    print(gauss_res)
    print("\nSolution:\n", x)

    print("\nFinished")

    print("\nChecking if the answer is correct:")
    print("Ax should be equal to initial f:")
    test = np.isclose(matrix@x, f)
    print(f"(matrix @ x)[i] == f[i]:\n{test}")

if __name__ == "__main__":
    main()
