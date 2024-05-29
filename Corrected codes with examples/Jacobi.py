import numpy as np
from numpy.linalg import norm
from matrix_utility import is_diagonally_dominant



def swap_row(mat, i, j):
    N = len(mat)
    for k in range(N + 1):
        temp = mat[i][k]
        mat[i][k] = mat[j][k]
        mat[j][k] = temp


def get_best_diagonal(mat, n):
    for i in range(n):
        pivot_row = i
        v_max = mat[pivot_row][i]
        for j in range(i + 1, n):
            if abs(mat[j][i]) > abs(v_max):
                v_max = mat[j][i]
                pivot_row = j

        # if a principal diagonal element is zero,it denotes that matrix is singular,
        # and will lead to a division-by-zero later.
        if not mat[pivot_row][i]:
            return i  # Matrix is singular

        # Swap the current row with the pivot row
        if pivot_row != i:
            swap_row(mat, i, pivot_row)
    return -1


def check_if_singular(mat, b, n):
    singular_flag = get_best_diagonal(mat, n)
    if singular_flag != -1:
        if b[singular_flag]:
            print("Singular Matrix (Inconsistent System)")
            return 0
        else:
            print("Singular Matrix (May have infinitely many solutions)")
            return 0
    return -1



def get_D(mat, n):
    D = np.zeros((n, n), dtype=np.double)
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = mat[i][j]
    return D


def get_L(mat, n):
    L = np.zeros((n, n), dtype=np.double)
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = mat[i][j]
    return L


def get_U(mat, n):
    U = np.zeros((n, n), dtype=np.double)
    for i in range(n):
        for j in range(n):
            if i < j:
                U[i][j] = mat[i][j]
    return U


def get_jacobi_H(mat, n):
    return np.linalg.inv(get_D(mat, n))


def get_jacobi_G(mat, n):
    inverse_of_D = np.linalg.inv(get_D(mat, n))
    L_plus_U = get_L(mat, n) + get_U(mat, n)
    return np.dot(inverse_of_D, L_plus_U)*(-1)


def jacobi_iterative(mat, b, n, X0, TOL=0.001):
    H = get_jacobi_H(mat, n)
    G = get_jacobi_G(mat, n)
    k = 1
    print("Jacobi:")
    print("Iteration" + "\t\t\t".join(
        [" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(mat) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    while True:
        x = np.dot(G, X0) + np.dot(H, b)
        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))
        if norm(x - X0, np.inf) < TOL:
            return tuple(x)
        k += 1
        X0 = x.copy()


def get_jacobi_solution(mat, b, n, X0):
    if check_if_singular(mat, b, n) == -1:
        if not is_diagonally_dominant(mat):
            print('Matrix is not diagonally dominant!\n')
        else:
            return jacobi_iterative(mat, b, n, X0)


#########################################################


if __name__ == '__main__':
    mat = np.array([[3, -1, 1],
                    [0, 2, -1],
                    [0, 1, -2]])
    n = len(mat)
    b = np.array([4, -1, -3])
    x = np.zeros_like(b, dtype=np.double)

    solution = get_jacobi_solution(mat, b, n, x)
    print(solution, "\n")