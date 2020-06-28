def objective(vars):
    return 2*vars[0] - 3*vars[1] - 2*vars[2] - 3*vars[3]
    #return prices.T @ vars.reshape((-1,1))

def cons_f(vars):
    g = np.zeros([2, 1])
    g[0] = 2 - vars.sum()
    g[1] = 10 * vars[0] + 5 * vars[1] + 3 * vars[2] + 4 * vars[3] - 10
    #g = sizes.T @ vars.reshape((-1,1)) - 5
    return np.array(g).reshape((-1,))

import numpy as np
np.random.seed(0)
n_x = 1          # Number of continuous variables
n_y = 3          # Number of integer variables
N   = n_x + n_y  # Total number of variables
# prices = -np.random.rand(N, 1)
# sizes = np.random.rand(N, 1)
# ----Define the bounds for the n_x vars-------- #
lb = [0.]
ub = [np.inf]
# ---------------------------------------------- #

#lb = np.array([0.] + [0.]*3)  # lb on parameters (this is inside the exponential)
#ub = np.array([3000] + [1.]*3)  # lb on parameters (this is inside the exponential)
OPT = BB(N=N, n_y=n_y, objective=objective, constraints=cons_f,
         bounds=[lb, ub])


bestres, bestnode = OPT.bbsolve()
print(x)
