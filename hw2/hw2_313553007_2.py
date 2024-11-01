import numpy as np 
import argparse


def combination(N, num_1):
    tmp = num_1 if N - num_1 > num_1 else N -num_1
    denominator = 1
    for i in range(1, N-tmp+1):
        denominator = denominator * i

    nominator = 1
    for i in range(tmp+1, N+1):
        nominator = nominator * i
    return nominator / denominator    
    
class BetaBinomial:
    def __init__(self, a, b):
        self.a = a  # 1
        self.b = b  # 0
    def update_parameters(self, num_1, num_0):
        self.a = self.a + num_1
        self.b = self.b + num_0
    def likelihood(self, p, N, num_1):
        return combination(N, num_1)*(p**num_1) * ((1-p)**(N-num_1))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(prog="Machine learning homework2 for Online learning")
    parser.add_argument('-a', type=float, required=True, help="Initial paramterer for a.")
    parser.add_argument('-b', type=float, required=True, help="Initial paramterer for b.")
    parser.add_argument('--file', type=str, help="File saving test data", default="data/testfile.txt")
    args = parser.parse_args()
    a, b = args.a, args.b
    TEST_FILE = args.file
    model = BetaBinomial(a,b)
    # print(combination(6 ,3 ))
    # exit()
    with open(TEST_FILE, 'r') as file:
        data = [ line.strip() for line in  file.readlines()]
    
    for i, data in enumerate(data, 1):
        N = len(data)
        num_1 = data.count("1")
        p = num_1 / N # * MLE of Binomial trial
        likelihood = model.likelihood(p, N, num_1)
        
        print(f"Case {i}: {data}")
        print(f"Likelihood: {likelihood}")
        print(f"Beta Prior:     a={model.a}, b={model.b}")
        model.update_parameters(num_1, N-num_1)
        print(f"Beta posterior: a={model.a}, b={model.b}")
        print()