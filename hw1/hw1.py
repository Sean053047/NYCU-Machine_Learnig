
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(10)
class Matrix:
    @staticmethod
    def identity(n:int):
        ii = np.zeros((n,n), dtype=np.float64)
        for ni in range(n):
            ii[ni,ni] = 1.0
        return ii
    
    @staticmethod
    def transpose(arr:np.ndarray):
        '''Get the tranpose of given 2D array.'''
        if len(arr.shape) != 2:
            raise "The array for transposing should be 2D array."
        result = np.empty((arr.shape[1], arr.shape[0]), dtype=arr.dtype)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i][j] = arr[j][i]
        return result
    
    @staticmethod
    def product(a:np.ndarray, b:np.ndarray):
        if a.shape[1] != b.shape[0]:
            raise "Wrong shape for Matrix multiplication."
        result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                result[i,j] = sum(a[i,:] * b[:, j])        
        # print("Product->",np.all(np.abs(result - a@b)< 1e-4))
        
        return result
    
    @classmethod
    def inverse(cls, arr:np.ndarray):
        P, L ,U = cls.LU_decomposition(arr)
        '''
        arr @ inverse(arr) = I
        arr = P @ L @ U
        transpose(P) = inverse(P)
        Assume U @ inverse(arr) = Y
        step 1: Use L @ Y = inverse(P) @ I to get Y.
        step 2: Use U @ inverse(arr) = Y to get inverse(arr)
        '''
        n = len(arr)
        # Step 1: Solve Y
        var = cls.product(P, np.identity(n))
        Y = np.zeros((n,n))
        for ni in range(n):
            div_L = L[[ni], :ni]
            div_Y = Y[:ni,:]
            Y[ni, :] = (var[ni,:] - cls.product(div_L, div_Y)) / L[ni,ni]
        # Step 2: Solve inverse(arr)
        R = np.zeros((n,n))
        for ni in range(n-1, -1, -1):
            div_U = U[[ni], ni:]
            div_R = R[ni: ,:]
            R[ni,:] = (Y[ni, :] - cls.product(div_U, div_R)) / U[ni,ni]
        # assert np.all(np.abs(np.linalg.inv(arr) - R) < 1e-8), "Warning for inverse."
        return R 
    @classmethod
    def LU_decomposition(cls, arr:np.ndarray):        
        n = len(arr)
        P = cls.identity(n)
        L = cls.identity(n)
        U = cls.identity(n)
        # Swap to make the first element of arr is the maximum in that column.
        max_indx = sorted([i for i in range(arr.shape[0])], key=lambda indx: arr[indx,0])[-1]
        P[[0, max_indx],:] = P[[max_indx, 0], :]
        tmp_arr = cls.product(P, arr)
        
        for ni in range(0,n):
            # Solve U matrix first for each stage n_indx.
            div_arr = tmp_arr[[ni], ni:]
            div_L = L[[ni], :ni]
            div_U = U[:ni , ni:]
            U[ni, ni:] = div_arr - cls.product(div_L, div_U)
            
            # Solve L matrix first for each stage n_indx.
            div_arr =  tmp_arr[ni:, [ni]]
            div_L = L[ni:, :ni]
            div_U = U[:ni, [ni]]
            L[ni: , [ni]] = (div_arr - cls.product(div_L, div_U)) / U[ni,ni]
        # from scipy.linalg import lu
        # p,l,u = lu(arr)
        # assert np.all(np.abs(np.stack((p,l,u), axis=2) - np.stack((P,L,U), axis=2)) < 0.0001), "Warning for P,L,U values."
        return P, L, U

    @staticmethod
    def sign(arr:np.ndarray):
        s = np.ones_like(arr)
        for i in range(arr.size):
            s[i] = 1 if arr[i] >=0 else -1
        return s     

    @staticmethod
    def norm(arr:np.ndarray, norm=2):
        result = 0
        for i in range(arr.size):
            if norm ==1:
                result = result + abs(arr[i])
            else:
                result = result +arr[i]**norm
        result = result ** (1/norm)
        return result[0]
# Alias of Matrix
class M(Matrix):...

def read_file(data:str):
    with open(data,'r') as f:
        data = f.readlines()
        data = [ dd.strip().split(',') for dd in data]
    X = np.array([ float(d[0]) for d in data])
    Y = np.array([ float(d[1]) for d in data])
    return X, Y

def plot_result(A:np.ndarray, Y:np.ndarray, F:np.ndarray,title='figure'):
    x,y = A[:,1], Y[:]
    predict_y = M.product(A,F)
    total_error = LSE(A,F,Y)
    print(f"{title}:")
    for i,f in enumerate(F):
        if i ==0:
            print(f"({f[0]})", end='')
        else:
            print(f"({f[0]}x^{i})", end='')
        if i != len(F)-1:
            print(" + ", end='')
        else:
            print()
    print("Total error: ", total_error, '\n')
    
    plt.plot(x,y ,'o', label="Data points")
    plt.plot(x,predict_y, label='Prediction')
    plt.title(title + f"| LSE: {total_error}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

def create_feature_array(X:np.ndarray, poly_bases:int):
    feature = [ X**i for i in range(poly_bases)]
    return M.transpose(np.array(feature, dtype=np.float64))

def LSE(A:np.ndarray, X:np.ndarray, Y, get_mean:bool = False):
    if len(X.shape) == 1:
        X = X.reshape((-1, 1))
    if len(Y.shape) == 1:
        Y = Y.reshape((-1,1))
    lse = sum( (M.product(A, X) - Y) **2 )
    if get_mean: 
        num_data = A.shape[0]
        lse = lse / num_data
    return lse[0]

def closed_form_LSE(A:np.ndarray, Y:np.ndarray, r_lambda:float):
    Y = Y.reshape([-1,1]) if len(Y.shape) == 1 else Y
    AT = M.transpose(A)
    ATA = M.product(AT, A)
    _A = ATA + r_lambda * np.identity(ATA.shape[0])
    function_X = M.product(
                    M.product(M.inverse(_A), AT), Y
                )
    return function_X

def steepest_descent(A:np.ndarray, Y:np.ndarray, lrs:dict[int:float], penalty:float= 0.01, num_iter = 10000):
    """
    LS1+ L1-norm => | AX - Y |^2 + penalty * |X|
    To avoid gradient explosion...
        => 1. Scale learning rate by L2-norm of gradient.
        => 2. Use learning rate scheduling.
    """ 
    N = A.shape[1]
    Y = Y.reshape((-1,1))
    # * Based on Lasso, don't add penalty to bias term.
    penalty = penalty * np.ones((N, 1))
    penalty[0,0] = 0 
    # * Randomly generate function
    X = np.random.rand(N, 1)
    change_iter = list(lrs.keys())
    for i in range(num_iter):
        # learning scheduling
        if i in change_iter:
            lr =lrs[i]
            
        ATAX = M.product(
            M.product( 
                M.transpose(A), A
            ), X
        ) 
        ATY =  M.product(M.transpose(A), Y)
        gradient = 2*(ATAX - ATY) + penalty*M.sign(X)
        lr_tmp = lr / (0.01*(M.norm(gradient) + 1e-9))
        X = X - lr_tmp* gradient
        
    return X 
        

def Newtons_method(A:np.ndarray, Y:np.ndarray, num_iter=50):
    '''
    Goal: Find out the function to minimize LSE without regularization.
    gradient: 2ATAX - 2ATY
    hessian: 2ATA
    '''
    N = A.shape[1]
    Y = Y.reshape((-1,1))
    X = np.random.rand(N, 1)
    prior_LSE = 0 
    for i in range(num_iter):
        gradient = 2*( M.product(M.product(M.transpose(A), A), X) - M.product(M.transpose(A), Y) )
        hessian = 2*M.product(M.transpose(A), A)
        X = X - M.product( M.inverse(hessian), gradient)
        
        # * Set convergence condition
        current_LSE = LSE(A, X, Y)
        if abs(prior_LSE - current_LSE ) < 1e-10:
            break
        prior_LSE = current_LSE
    return X



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="Machine Learning Homework 1")
    parser.add_argument("--data", type=str, required=True, help="Data file to be processed.")
    parser.add_argument("--poly_bases", type=int, required=True, help="The number of polynomial bases.")
    parser.add_argument("--r_lambda", type=float, default=0.01, help="Parameter lambda only for LSE.")
    args = parser.parse_args()
    X, Y = read_file(args.data)
    
    lrs = {
        0 : 1e-6,
        100:1e-3,
        5000:5e-5,
        7000:5e-7
    }
    A = create_feature_array(X, args.poly_bases)
    F1 = closed_form_LSE(A, Y, args.r_lambda)
    F2 = steepest_descent(A, Y, lrs)
    F3 = Newtons_method(A, Y)
    
    plot_result(A, Y, F1, 'closed-form LSE')
    plot_result(A, Y, F2, 'steepest_descent')
    plot_result(A, Y, F3, "Newton's method")
    