import numpy as np
from scipy.linalg import block_diag


class SLS():
    """
    Class that contains SLS related functions.
    """

    def __init__(self, N, nx, nu):
        """
        Initialize the SLS class.
        :param N: int: The horizon length.
        :param nx: int: The state dimension.
        :param nu: int: The control dimension.
        """
        self.N = N
        self.nx = nx
        self.nu = nu
        self.nw = nx

    def get_submatrix_Phi_x(self, Phi_x, i, j):
        """
        Takes as input a matrix of size (N n_x, N n_x) and returns its submatrices Phi_x^i,j.
        :return:
        """
        # return Phi_x[i*self.nx:(i+1)*self.nx, j*self.nx:(j+1)*self.nx]
        raise NotImplementedError

    def get_submatrix_Phi_u(self, Phi_u, i, j):
        """
        Takes as input a matrix of size (N n_x, N n_u) and returns its submatrices Phi_u^i,j.
        :return:
        """
        # return Phi_u[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu]
        raise NotImplementedError

    @staticmethod
    def eval_cost(N, Q, R, Q_f, Phi_x_mat, Phi_u_mat):
        Q_blk = block_diag(np.kron(np.eye(N), Q), Q_f)
        R_blk = np.kron(np.eye(N), R)
        Phi_mat = np.vstack([Phi_x_mat, Phi_u_mat])
        # replace nan by zeros
        Phi_mat = np.nan_to_num(Phi_mat)

        return np.linalg.norm(block_diag(Q_blk, R_blk) @ Phi_mat, ord='fro')

    @staticmethod
    def convert_tensor_to_matrix(tensor):
        """
        Convert a tensor to a matrix.
        :param tensor: tensor: The tensor to convert.
        :return: numpy array: The matrix.
        """
        # extract tensor size
        size = tensor.shape
        N = size[0]
        M = size[1]
        n = size[2]
        m = size[3]

        # return the corresponding projected matrix
        return tensor.transpose(0, 2, 1, 3).reshape(N * n, M * m)

    @staticmethod
    def convert_matrix_to_tensor(matrix, horizon, a, b):
        """
        Convert a matrix to a tensor.
        :param matrix: numpy array: The matrix to convert.
        :param horizon: int: The horizon length.
        :param a: int: The first dimension of the tensor.
        :param b: int: The second dimension of the tensor.
        :return: tensor: The tensor.
        """
        return matrix.reshape(horizon, a,horizon , b).transpose(0, 2, 1, 3)
        #test: np.allclose(Phi_xx_mat, SLS.convert_tensor_to_matrix(Phi_xx_mat.reshape(N+1, nx, N+1,nx).transpose(0,2,1, 3) ))

    @staticmethod
    def convert_tensor3_to_matrix(tensor):
        """
        Convert a tensor to a matrix.
        :param tensor: tensor: The tensor to convert.
        :return: numpy array: The matrix.
        """
        # extract tensor size
        size = tensor.shape
        N = size[0]
        M = size[1]
        n = size[2]

        # return the corresponding projected matrix
        return tensor.transpose(0, 2, 1).reshape(N * n, M)

    @staticmethod
    def convert_list_to_blk_matrix(A_list):
        """
        Convert a list of matrices to a block matrix.
        :param append_zero:
        :param A_list: list: The list of matrices.
        :return: numpy array: The block matrix.
        """
        # extract the size of the matrices
        n = A_list[0].shape[0]
        m = A_list[0].shape[1]

        N = len(A_list)

        # initialize the block matrix
        A_blk = np.zeros((n * N, m * N))

        # fill the block matrix
        for i, A in enumerate(A_list):
            A_blk[i * n:(i + 1) * n, i * m:(i + 1) * m] = A

        return A_blk

    @staticmethod
    def get_block_downshift_matrix(N, n):
        """
        Get the block downshift matrix.
        :param n: int: The size of the block.
        :param N: int: The number of blocks.
        :return: numpy array: The block downshift matrix.
        """
        # initialize the block downshift matrix
        D = np.zeros((n * N, n * N))

        # fill the block downshift matrix
        for i in range(N - 1):
            D[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = np.eye(n)

        return D
