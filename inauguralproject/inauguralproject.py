from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar



class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)
        

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1


        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0: 
            H = np.minimum(HM,HF)
        else:
            H =((1-par.alpha)*HM**((par.sigma-1)/par.sigma+1e-10)+par.alpha*HF**((par.sigma-1)/par.sigma+1e-10))**((par.sigma/(par.sigma+1e-10-1))) 

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]
        
        

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
        

    def solve_continue(self,do_print=False):
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # objective function to minimize ( minimize the negative of the objective function to maximize utility)
        obj_func = lambda x: -self.calc_utility(x[0], x[1], x[2], x[3])

        # constraints (time constraints for both partners)
        cons = ({'type': 'ineq', 'fun': lambda x: 24 - x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x: 24 - x[2] - x[3]})

        # bounds (discrete bounds between 0 to 24)
        bounds = [(0, 24), (0, 24), (0, 24), (0, 24)]

        # initial guess
        x0 = np.array([12, 12, 12, 12])

        # solve using Nelder-Mead method
        res = optimize.minimize(obj_func, x0, bounds=bounds, constraints=cons, method='Nelder-Mead')

        # store results
        opt.LM, opt.HM, opt.LF, opt.HF = res.x

        # print results
        if do_print:
            for k, v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
  

    def run_regression(self):
        # ensure that the model is solved before running the regression
        par = self.par
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF
            sol_ = self.solve_continue() # or self.solve_discrete()
            self.sol.LM_vec[i], self.sol.HM_vec[i], self.sol.LF_vec[i], self.sol.HF_vec[i] = sol_.LM, sol_.HM, sol_.LF, sol_.HF

        y = np.log(self.sol.HF_vec/self.sol.HM_vec)
        x = np.log(par.wF_vec/par.wM)
        A = np.vstack([np.ones(x.size),x]).T
        self.sol.beta0, self.sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        return self.sol

    def utility_nonconstantwages(self, LM, HM, LF, HF):
        """Calculate utility"""
        
        par = self.par
        sol = self.sol
        
        # a. consumption of market goods
        C = par.wM * LM + par.wF * LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1 - par.alpha) * HF**par.alpha
        elif par.sigma == 0:
            H = np.minimum(HM, HF)
        else:
            H = ((1 - par.alpha) * HM**((par.sigma - 1) / par.sigma + 1e-10) + par.alpha * HF**((par.sigma - 1) / par.sigma + 1e-10))**((par.sigma / (par.sigma + 1e-10 - 1)))

        # c. total consumption utility
        Q = C**par.omega * H**(1 - par.omega)
        utility = np.fmax(Q, 1e-8)**(1 - par.rho) / (1 - par.rho)

        # d. disutility of work
        epsilon_ = 1 + 1 / par.epsilon
        TM = LM + HM
        TF = LF + HF
        disutility = par.nu * (TM**epsilon_ / epsilon_ + TF**epsilon_ / epsilon_)
        
        # e. adjusting wages based on hours worked
        adjusted_wF = par.wF * LF / (LF + HF)
        adjusted_wM = par.wM * LM / (LM + HM)

        # f. update market consumption with adjusted wages
        C = adjusted_wM * LM + adjusted_wF * LF

        # g. recalculate total consumption utility
        Q = C**par.omega * H**(1 - par.omega)
        utility = np.fmax(Q, 1e-8)**(1 - par.rho) / (1 - par.rho)

        # Adjusting wages
        adjusted_result = utility - disutility

        return adjusted_result


    def solve_continuous_nonconstantwages(self, do_print=False):
        """ solve model continuously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # objective function to minimize (minimize the negative of the objective function to maximize utility)
        obj_func = lambda x: -self.calc_utility(x[0], x[1], x[2], x[3])

        # constraints (time constraints for both partners)
        cons = [{'type': 'ineq', 'fun': lambda x: 24 - x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x: 24 - x[2] - x[3]}]

        # bounds (discrete bounds between 0 to 24)
        bounds = [(0, 24), (0, 24), (0, 24), (0, 24)]

        # initial guess
        x0 = np.array([12, 12, 12, 12])

        # solve using Nelder-Mead method
        res = optimize.minimize(obj_func, x0, bounds=bounds, constraints=cons, method='Nelder-Mead')

        # store results
        opt.LM, opt.HM, opt.LF, opt.HF = res.x

        # print results
        if do_print:
            for k, v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt


    def solve_nonconstantwF_vec(self, wflist):
        lnwFwM_vec = np.empty(len(wflist))
        lnHFHM_vec = np.empty(len(wflist))

        for i, wf1 in enumerate(wflist):
            self.par.wF = wf1
            lnwFwM = np.log(self.par.wF / self.par.wM)
            result = self.solve_continuous_nonconstantwages()
            lnHFHM = np.log(result.HF / result.HM)
            lnwFwM_vec[i] = lnwFwM
            lnHFHM_vec[i] = lnHFHM

        return lnwFwM_vec, lnHFHM_vec


    def run_regression_nonconstantwages(self):
        sigma_list = np.linspace(0.8, 1.2, 11)

        best_res = np.inf
        best_sigma = None
        best_beta0 = None
        best_beta1 = None
        # Run through the question 3 for-loop with varying sigma values
        for s in sigma_list:
            self.par.sigma = s
            wflist = (0.8, 0.9, 1.0, 1.1, 1.2)

            lnwFwM_vec, lnHFHM_vec = self.solve_nonconstantwF_vec(wflist)

            # Run the regression to estimate the coefficients beta0 and beta1 that minimize the sum of squared errors
            x = lnwFwM_vec
            y = lnHFHM_vec
            A = np.vstack([np.ones(x.size), x]).T
            self.sol.beta0, self.sol.beta1 = np.linalg.lstsq(A, y, rcond=None)[0]

            # Calculate the sum of squared errors, new_res based on the estimated coefficients
            new_res = (0.4 - self.sol.beta0) ** 2 + (-0.1 - self.sol.beta1) ** 2

            # Check if the sum of squared errors is smaller than the previous best estimate
            if new_res < best_res:
                best_res = new_res
                best_sigma = s
                best_beta0 = self.sol.beta0
                best_beta1 = self.sol.beta1

        # Print the best results
       # print(f"Best Sigma: {best_sigma}")
       # print(f"Best Beta0: {best_beta0}")
       # print(f"Best Beta1: {best_beta1}")
       # print(f"Best Residual: {best_res}")
