import numpy as np 
from libsvm.svmutil import *
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import json
from tqdm import tqdm
# * Gaussian Process
class GaussianProcess:
    def __init__(self, kernel_attr:dict, beta):
        self.kernel_attr = kernel_attr
        self.beta = beta
        self.C:np.ndarray = np.empty(0)
        self.train_data:np.ndarray = np.empty(0)
        self.train_label: np.ndarray = np.empty(0)
        
    @property
    def N(self):
        return len(self.train_data)       
    
    def RQ_kernel(self, src, tgt):
        '''
        Rational Quadratic kernel
        src: 1*N matrix.\n
        tgt: 1*M matrix.\n
        return: 
        K: N*M matrix  (number_src, number_tgt)
        '''
        var = self.kernel_attr.get('var')
        alpha = self.kernel_attr.get('alpha')
        l = self.kernel_attr.get('l')
        src= src.reshape( -1, 1)
        tgt = tgt.reshape( -1, 1)
        K = var * ( 1+ cdist(src, tgt, 'sqeuclidean') / (2 * alpha* np.square(l))) **(-alpha)
        return K
    
    def fit(self, train_data:np.ndarray, train_label:np.ndarray):
        self.train_data = train_data
        self.train_label = train_label
        self.C = self.RQ_kernel(self.train_data, self.train_data) + 1/ self.beta * np.identity(self.N)
        
    def predict(self, test_data:np.ndarray):
        M = len(test_data)
        y = self.train_label.reshape(-1, 1)
        KK = self.RQ_kernel(test_data, self.train_data)                     # * K(x', x ), M*N matrix
        u = KK@ np.linalg.inv(self.C) @ y
        K_own = self.RQ_kernel(test_data, test_data) + 1/self.beta*np.identity(M) # * K(x', x'), M matrix
        var = np.diag( K_own -  KK @ np.linalg.inv(self.C) @ KK.T)
        return u.flatten(), var.flatten()
        
    def plot_results(self, test_data, test_mean, test_var):
        plt.plot( test_data, test_mean, color='red')
        plt.fill_between(test_data, test_mean+1.96*test_var, test_mean-1.96*test_var)
        plt.scatter(self.train_data, self.train_label, color='black', label="training data")
        plt.title("Gaussian Process\n"\
                  "RQ params: alpha: {:.2f}, l: {:.2f}, var: {:.2f}".format(self.kernel_attr['alpha'], self.kernel_attr['l'], self.kernel_attr['var']))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(-20,20)
        plt.show()
        
def gaussian_process():
    global data, beta, args
    if args.optimize:
        initial_guess = [1.0, 1.0, 1.0]
        bounds = [(1e-5, None), (1e-5, None), (1e-5, None)]
        result = minimize(gp_neg_log_likelihood, initial_guess, bounds=bounds)
        if not result.success:
            print("Failed to find optimal parameters.")
            exit()
        var, alpha, l = result.x
        print("Succeed to fine optimal solution with negative likelihood: {}".format(result.fun))
        print("Kernel Parameters: \nvar: {}, \nalpha: {}, \nl: {}".format(var, alpha, l))
    else:
        var = args.var
        alpha = args.alpha
        l = args.l 
    kernel_attr = {'var':var, 'alpha':alpha, 'l':l}
    gp =GaussianProcess(kernel_attr, beta)
    gp.fit(data[:,0], data[:,1])
    pred_data = np.arange(-60, 61,0.1)
    pred_u, pred_var = gp.predict(pred_data)
    gp.plot_results(pred_data, pred_u, pred_var)
    
def gp_neg_log_likelihood(params):
    var, alpha, l = params
    global data, beta
    kernel_attr = {'var': var, 'alpha':alpha, 'l':l}
    gp = GaussianProcess(kernel_attr, beta)
    x, y = data[:,0], data[:,1]
    gp.fit(x, y)
    pred_u, pred_var = gp.predict(x)
    J = np.sum( -0.5*np.log(2*np.pi * pred_var)- 0.5 *(y-pred_u) **2 / pred_var)
    return -J

class svm_utils:
    def __init__(self):
        self.record = defaultdict(list)
        
    @staticmethod
    def make_args( kernel, *args, **kwargs):
        svm_args = ['-s', '0']
        if kernel == 'rbf':
            c = kwargs.get('c') if 'c' in kwargs else args[0]
            g = kwargs.get('g') if 'g' in kwargs else args[1]
            svm_args.extend( ['-t', '2', '-c', str(c), '-g', str(g)])
            return svm_args 
        elif kernel ==  'poly':
            c = kwargs.get('c') if 'c' in kwargs else args[0]
            g = kwargs.get('g') if 'g' in kwargs else args[1]
            r = kwargs.get('r') if 'r' in kwargs else args[2]
            d = kwargs.get('d') if 'd' in kwargs else args[3]
            svm_args.extend(['-t', '1', '-c', str(c), '-g', str(g), '-r', str(r), '-d', str(d)])        
            return svm_args    
        elif kernel == 'linear':
            c = kwargs.get('c') if 'c' in kwargs else args[0]
            svm_args.extend(['-t', '-0', '-c', str(c)])
            return svm_args 
        elif kernel == 'lrbf':
            c = kwargs.get('c') if 'c' in kwargs else args[0]
            g = kwargs.get('g') if 'g' in kwargs else args[1]
            svm_args.extend( ['-t', '4', '-c', str(c), '-g', str(g)])
            return svm_args 

    @staticmethod
    def args_to_dict(svm_args:list):
        out = dict()
        for i, arg in enumerate(svm_args):
            if 'c' in arg:
                out['c'] = float(svm_args[i+1])
            elif 'g' in arg:
                out['g'] = float(svm_args[i+1])
            elif 'r' in arg:
                out['r'] = float(svm_args[i+1])
            elif 'd' in arg:
                out['d'] = float(svm_args[i+1])
        return out
            
    def get_best_params(self, kernel):
        svm_args = max(self.record[kernel], key=lambda a: a[1])[0]
        params_best = {svm_args[i-1].strip('-'):svm_args[i] 
                    for i in range(1, len(svm_args)) if svm_args[i-1] in ('-g', '-r', '-d', '-c')
                }
        return params_best
    
    def search_best_param(self, y_train, x_train, kernel, num_folds:int):
        print(f"Grid Search hyperparameters of {kernel} kernel with {num_folds}-fold cross validation")
        stop_iter_ratio = 0.2
        # * Best: C = 0.03
        if kernel == "linear":
            params_ranges = { 'c': np.arange(0.01, 3, step=0.01),}
        elif kernel == 'poly':
            params_ranges = {
                'c': np.arange(0.01, 15, step=0.5),
                'g': np.arange(0.01, 10, step=0.2),
                'r': np.arange(-10, 10, step=0.2),
                'd': np.arange(1, 20 , step=1)
            }
        elif kernel == 'rbf':
            params_ranges = {
                'c': np.arange(0.01, 10, step=0.5),
                'g': np.arange(0.01, 10, step=0.03)
            }
        elif kernel == 'lrbf':
            params_ranges = {
                'c': np.arange(0.01, 3, step=0.01),
                'g': np.arange(0.01, 5, step=0.02)
            }
        def coordinate_search():
            nonlocal self, y_train, x_train, kernel, num_folds, params_ranges
            params_best ={ k: v[0] for k,v in params_ranges.items()}
            for iter in range(3):
                # Use the idea of coordinate descent.
                # Go through individual parameters. Optimize individually.
                for param, param_range in  params_ranges.items():
                    print(f"Optimize {param}, best_params: {params_best}")
                    if iter != 0:
                        params_best = self.get_best_params(kernel)
                        print("Reassign params_best:", params_best)
                    max_acc, lower_count = 0.0, 0     
                    tmp_params = params_best
                        
                    for val in param_range:
                        tmp_params[param] = val                    
                        svm_args = self.make_args(kernel, **tmp_params) + ['-v', str(num_folds), '-q']
                        _feature = x_train if kernel != 'lrbf' else self.linear_rbf(x_train, x_train, tmp_params['g'])
                        acc = svm_train(y_train, _feature, svm_args)
                        self.record[kernel].append((svm_args, acc))
                        
                        if acc> max_acc:
                            max_acc = acc
                            lower_count = 1
                        else:
                            lower_count += 1
                        if lower_count > stop_iter_ratio *len(param_range) or acc < max_acc *0.8:
                            break
                        print(svm_args, lower_count, stop_iter_ratio *len(param_range))
        
        def grid_search():
            nonlocal self, y_train, x_train, kernel, num_folds, params_ranges
            if kernel == 'linear':
                all_params = [ {'c':v } for v in params_ranges['c']]
            elif kernel == 'rbf' or kernel == 'lrbf':
                all_params = [ {'c':vc, 'g':vg} for vc in params_ranges['c'] for vg in params_ranges['g']]
            
            for tmp_params in all_params:
                svm_args = self.make_args(kernel, **tmp_params) + ['-v', str(num_folds), '-q']
                _feature = x_train if kernel != 'lrbf' else self.linear_rbf(x_train, x_train, tmp_params['g'])
                acc = svm_train(y_train, _feature, svm_args)
                self.record[kernel].append((svm_args, acc))
        
        if kernel =='poly':
            coordinate_search()
        else:
            grid_search()
        return max(self.record[kernel], key=lambda a: a[1])
        
    @staticmethod
    def linear_rbf(src1, src2, g):
        out = np.dot(src1, src2.T) + np.exp( -g * cdist(src1, src2, 'sqeuclidean'))
        out = np.concatenate((
            np.arange(1, len(out)+1).reshape(-1, 1), 
            out
        ), axis= 1)
        return out 

def svm():  
    global x_train, y_train, x_test, y_test, args
    svmu = svm_utils()
    num_folds = 5 
    # * Find best hyper parameters
    if args.optimize:
        json_data = dict()
        for kernel in [ 'linear', 'rbf', 'poly', 'lrbf']:
            best_args, best_acc = svmu.grid_search(y_train, x_train, kernel, num_folds)
            json_data[kernel] = svmu.args_to_dict(best_args).update({'best':best_acc})
        with open('best_params.json', 'w') as file:
            json.dump(json_data, file, indent=4)    
    else:
        params = { k:getattr(args, k) for k in ('c', 'g', 'r', 'd') if hasattr(args, k) and getattr(args, k) != None}
        svm_args = svmu.make_args(args.kernel, **params) +['-q']
        if args.kernel == 'lrbf':
            x_test = svmu.linear_rbf(x_test, x_train, params['g'])
            x_train = svmu.linear_rbf(x_train, x_train, params['g'])
        model = svm_train(y_train, x_train, svm_args)
        _, train_acc, _ = svm_predict(y_train, x_train, model, '-q')
        _, pred_acc, _ = svm_predict(y_test, x_test, model, '-q')
        print(f"Kernel {args.kernel}, params: {params}")
        print("Evaluation on Training Data => Accuracy:", train_acc[0])
        print("Evaluation on Test data => Accuracy:", pred_acc[0])
        
        
if __name__ == "__main__":
    # Input data
    import argparse
    parser = argparse.ArgumentParser(prog='ML hw5')
    parser.add_argument('mode', type=str, help='gp | svm')
    parser.add_argument('--optimize', action="store_true")
    # Gaussian Process
    parser.add_argument('--var', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('-l', type=float)
    # SVM
    parser.add_argument('--kernel', type=str, help="poly | linear | rbf | lrbf")
    parser.add_argument('-c', type=float)
    parser.add_argument('-g', type=float)
    parser.add_argument('-r', type=float)
    parser.add_argument('-d', type=int)
    args = parser.parse_args()
    if args.mode == 'gp':
        data_fpth = "data/input.data"
        with open(data_fpth) as file:
            data = np.array( [ tuple(map(float, dd.strip().split()))
                                for dd in file.readlines()],
                            dtype=np.float64)
        beta = 5
        gaussian_process()
    elif args.mode == 'svm':
        x_train = np.loadtxt('data/X_train.csv', delimiter=',', dtype=np.float64)
        y_train = np.loadtxt('data/Y_train.csv', delimiter=',', dtype=np.uint8)
        x_test = np.loadtxt('data/X_test.csv', delimiter=',', dtype=np.float64)
        y_test = np.loadtxt('data/Y_test.csv', delimiter=',', dtype=np.uint8)
        svm()