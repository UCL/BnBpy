import numpy as np
import scipy
import copy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import itertools
from heapq import *

counter = itertools.count()

class BB():

    def __init__(self, N, n_y, constraints, objective, bounds):#, optimization=[]):
        self.vars         = np.random.rand(N)
        self.constraints  = constraints
        self.ng           = constraints(self.vars).shape[0]
        self.objective    = objective
        self.bounds       = bounds
        self.bool_vars    = np.random.rand(n_y)
        self.children     = []
        lbi = [0.] * n_y
        ubi = [1.] * n_y
        self.bool_bounds  = [lbi, ubi]

    def is_integral(self):
        return all([abs(v - 1) <= 1e-3 or abs(v - 0) <= 1e-3 for v in self.bool_vars])

    def optimization1(self, lbi, ubi):
        objective = self.objective
        cons_f    = self.constraints
        vars      = self.vars
        bounds    = self.bounds
        ng        = self.ng
        lb = np.array([*bounds[0]] + [*lbi])  # lb on parameters (this is inside the exponential)
        ub = np.array([*bounds[1]] + [*ubi])  # lb on parameters (this is inside the exponential)
        bound = np.hstack((lb.reshape(-1, 1),
                            ub.reshape(-1, 1)))
        nonlinear_constraint = NonlinearConstraint(cons_f, np.array([-np.inf]*ng), np.array([0]*ng))
        res = minimize(objective, vars, method='SLSQP',
                       bounds=bound, constraints=nonlinear_constraint)
        opt = res.x
        optobj = res.fun
        return opt, optobj, res

    def branch(self):
        children = []
        for b in [0, 1]:
            n1 = copy.deepcopy(self)
            v = round(n1.heuristic())
            lbi = self.bool_bounds[0][:]#[0.] * np.shape(self.bool_vars)[0]
            ubi = self.bool_bounds[1][:]#[1.] * np.shape(self.bool_vars)[0]
            lbi[v] = b
            ubi[v] = b
            n1.children = []
            n1.bool_bounds = [lbi, ubi]
#            n1.bool_vars.remove(v)  # remove binary constraint from bool var set
 #           n1.vars.add(v)  # and add it into var set for later inspection of answer
            self.children.append(n1)   # eventually I might want to keep around the entire search tree. I messed this up though
            children.append(n1)
        return children

    def heuristic(self):
        # a basic heuristic of taking the ones it seems pretty sure about
        ni = np.shape(self.bool_vars)[0]
        v1 = np.ones(ni) * 50
        for i in range(ni):
            if abs(self.bool_bounds[0][i] - self.bool_bounds[1][i]) >1e-3:
                v1[i] = self.bool_vars[i]
        return min([(min(abs(1 - v), v), i, v) for i, v in enumerate(v1)])[1]

    def bbsolve(self):
        bools = self.bool_vars
        lbi = [0.]*np.shape(bools)[0]
        ubi = [1.]*np.shape(bools)[0]
        nx  = np.shape(self.bounds)[1]
        root = self
        _, res, detail = self.optimization1(lbi,ubi)#root.buildProblem().solve()
        heap = [(res, next(counter), root, (lbi, ubi))]
        bestres = 1e20  # a big arbitrary initial best objective value
        bestnode = root  # initialize bestnode to the root
        print(heap)
        nodecount = 0
        while len(heap) > 0:
            nodecount += 1  # for statistics
            print("Heap Size: ", len(heap))
            _, _, node, bool_bounds = heappop(heap)
            sols, res, detail  = node.optimization1(bool_bounds[0], bool_bounds[1])
            node.bool_vars = sols[nx:]
            node.solution  = sols
            print("Result: ", res)
            if detail.status == 0:
                if res > bestres - 1e-3:  # even the relaxed problem sucks. forget about this branch then
                    print("Fathom Relaxed Problem: Relaxed larger than Best-so-far.")
                    pass
                elif node.is_integral():
                    print("New Best-so-far found.")
                    bestres  = res
                    bestnode = node
                else:
                    new_nodes = node.branch()
                    for new_node in new_nodes:
                        heappush(heap, (res, next(counter),
                                        new_node, new_node.bool_bounds))
            else:
                print("Fathom Relaxed Problem: Infeasible solution found(It may be numerical issue).")
        print("Nodes searched: ", nodecount)
        return bestres, bestnode

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
lbi = [0.] * n_y
ubi = [1.] * n_y
x,y, detail = OPT.optimization1(lbi, ubi)
bestres, bestnode = OPT.bbsolve()
print(x)