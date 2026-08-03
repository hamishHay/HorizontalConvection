import numpy as np
from scipy.optimize import root_scalar


def get_eqlb_z(θm, Ra, beta=0.25, gamma=0.27):
    """
    Solve for the equilibrium z using bisection.
    """

    def f(z):
        return gamma * Ra**beta * (1 - θm)**(1 + beta) * z**(3*beta - 1) * (1.0 - z) - θm
        
    sol = root_scalar(f, bracket=[1e-10, 1.0 - 1e-10], method="bisect")

    if not sol.converged:
        raise RuntimeError("Root finding failed.")

    return sol.root

def optimum_m(Ra, Htilde, a=1.3, b=3.52):
    # Htilde  = 1 - get_eqlb_z(θm, Ra)
    E = 0.14 * Ra**0.72
    Hups = Htilde * np.sqrt(E)

    return a*Hups + b


def optimum_beta(Ra, θm, eps, c=0.435, m=8.5):
    beta_opt = 1.51

    # Step 1: Get zinf
    zinf = get_eqlb_z(θm, Ra, beta=0.25, gamma=0.27)

    # Step 2: Get effective Rayleigh
    Ra_eff = Ra * zinf**3 * (1.0 - θm)

    # Step 3: Check if near turbulence transition and, if so,
    #         update the effective Rayleigh number
    if Ra_eff >= 1e8:
        zinf = get_eqlb_z(θm, Ra, beta=1/3, gamma=0.07)
        Ra_eff = Ra * zinf**3 * (1.0 - θm)

    # Step 4: Obtain strain rate using E--Ra_eff relationsips
    E = 0.14 * Ra_eff**0.72 # Should this change for turbulence?

    m = optimum_m(Ra, θm, 1 - zinf)
    # Step 5: Get optimum beta 
    beta_opt = 1.0/np.sqrt(m * eps * np.sqrt(E) + c)

    # Question: Does m depend on H?
    return beta_opt, zinf

def get_E_RaF(RaF): # Currently only laminar
    return 0.32 * RaF**0.57

def optimum_m_RaF(Hups, a=1.3, b=3.52):
    # Htilde  = 1 - get_eqlb_z(θm, Ra)

    return a*Hups + b



def optimum_beta_RaF(RaF, θm, eps, c=0.435, m=8.5):
    beta_opt = 1.51

    RaF_eff = RaF * (1- θm)**4
    
    # Step 4: Obtain strain rate using E--Ra_eff relationsips
    E = get_E_RaF(RaF_eff) # Should this change for turbulence?
    
    
    Hups = θm * np.sqrt(E) # Rescaled ice thickness

    m = optimum_m_RaF(Hups)

    # Step 5: Get optimum beta 
    beta_opt = 1.0/np.sqrt(m * eps * np.sqrt(E) + c)

    # Question: Does m depend on H?
    return beta_opt

