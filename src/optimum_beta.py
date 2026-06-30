import numpy as np
from scipy.optimize import root_scalar


def get_eqlb_z(θm, Ra, beta=0.25, gamma=0.27):
    """
    Solve for the equilibrium z using bisection.
    """

    def f(z):
        return gamma * Ra**beta * (1 - θm) ** (1.0 + beta) * z ** (3*beta - 1) * (1.0 - z) - θm
        

    sol = root_scalar(f, bracket=[1e-10, 1.0 - 1e-10], method="bisect")

    if not sol.converged:
        raise RuntimeError("Root finding failed.")

    return sol.root


def optimum_beta(Ra, θm, eps, c=0.44, m=8.5):
    beta_opt = 1.51

    # Step 1: Get zinf
    zinf = get_eqlb_z(θm, Ra, beta=0.25, gamma=0.27)

    # Step 2: Get effective Rayleigh
    Ra_eff = zinf**3 * (1.0 - θm)

    # Step 3: Check if near turbulence transition and, if so,
    #         update the effective Rayleigh number
    if Ra_eff >= 1e8:
        zinf = get_eqlb_z(θm, Ra, beta=1/3, gamma=0.07)
        Ra_eff = zinf**3 * (1.0 - θm)

    # Step 4: Obtain strain rate using E--Ra_eff relationsips
    E = 0.14 * Ra_eff**0.72

    # Step 5: Get optimum beta 
    beta_opt = 1.0/np.sqrt(m * eps * np.sqrt(E) + 0.44)

    return beta_opt

