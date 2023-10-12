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

# ===========< Gauss >=============
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

# =======< LU-decomposition >==========
def LU(matrix, f):
    print("LU-decomposition")

    P, L, U = sci.linalg.lu(matrix)
    # print("L = \n", L)
    # print("U = \n", U)

    y = np.copy(f)
    L_res = gauss_direct(L, y)

    # print("L after gauss:\n", L)
    # print("y after gauss:\n", y)

    U_res = gauss_inverse(U, y)
    # print("U after gauss inverse:\n", U)
    # print("y after gauss inverse:\n", y)

    print("\nSolution:\n", y)
    return y

def mpi(matrix, f, B, F, x0):
    err = np.linalg.norm(matrix@x0 - f)
    errors = []

    while err > 1e-9:
        x0 = -B@x0 + F
        err = np.linalg.norm(matrix@x0 - f)
        # print(err)
        errors.append(err)

    return x0, errors

# ==========< Gauss-Seidel >============
def seidel(matrix, f):
    print("Seidel Method")
    x = np.copy(f) / 20

    L = np.copy(matrix)
    U = np.zeros(matrix.shape)
    
    for row in range(len(f)):
        for col in range(row + 1, len(f)):
            L[row][col] = 0
            U[row][col] = matrix[row][col]

    # print("seidel L =\n", L)
    # print("seidel U =\n", U)

    B = np.linalg.inv(L)@U
    F = np.linalg.inv(L)@f

    # print("B = L^-1 @ U:\n", B)

    x, errors = mpi(matrix, f, B, F, x)

    plt.yscale('log')
    plt.grid()
    plt.title("Seidel Method Error")
    plt.ylabel("Error")
    plt.xlabel("Step")

    plt.plot(errors, '.b-')
    plt.savefig("SeidelErrors.jpg")
    plt.clf()

    print("Seidel Method errors plotted")

def jacobi(matrix, f):
    print("Jacobi method")
    x = np.copy(f)

    D = np.zeros(matrix.shape)
    Oth = np.copy(matrix)

    for i in range(len(f)):
        D[i][i] = matrix[i][i]
        Oth[i][i] = 0

    B = np.linalg.inv(D) @ Oth
    F = np.linalg.inv(D) @ f

    x, errors = mpi(matrix, f, B, F, x)

    plt.yscale('log')
    plt.grid()
    plt.title("Jacobi Method Error")
    plt.ylabel("Error")
    plt.xlabel("Step")

    plt.plot(errors, '.b-')
    plt.savefig("JacobiErrors.jpg")
    plt.clf()

    print("Jacobi Method errors plotted")

def relax(matrix, f):
    print("Relaxation Method")
    x = np.copy(f) * 20

    L = np.copy(matrix)
    D = np.zeros(matrix.shape)
    U = np.zeros(matrix.shape)
    
    for row in range(len(f)):
        for col in range(row + 1, len(f)):
            L[row][col] = 0
            U[row][col] = matrix[row][col]
        D[row][row] = matrix[row][row]
        L[row][row] = 0

    print("Relax L =\n", L)
    print("Relax D =\n", D)
    print("Relax U =\n", U)

    w = 1.2

    B = np.linalg.inv(D + w*U) @ ((w-1)*D + w*L)
    F = w*np.linalg.inv(D + w*U) @ f

    x, errors = mpi(matrix, f, B, F, x)

    plt.yscale('log')
    plt.grid()
    plt.title("Relaxation Method Error")
    plt.ylabel("Error")
    plt.xlabel("Step")

    plt.plot(errors, '.b-')
    plt.savefig("RelaxationErrors.jpg")
    plt.clf()

    print("Relaxation Method errors plotted")

def main():
    np.set_printoptions(floatmode='maxprec', precision=3, suppress=True)
    matrix, f = gen_matrix(100)
    
    print("\nInput:")
    print(matrix)
    print(f)

    x = GaussSolution(matrix, f)
    y = LU(matrix, f)

    print("\nChecking if the Gauss answer is correct:")
    print("Ax should be equal to initial f:")
    test = matrix@x - f
    print(f"|(matrix @ x) - f)|:\n\t{np.linalg.norm(test)}")

    print("\nChecking if the LU answer is correct:")
    print(f"|(x - y)|:\n\t{np.linalg.norm(x - y)}")

    seidel(matrix, f)
    jacobi(matrix, f)
    relax (matrix, f)

if __name__ == "__main__":
    main()
