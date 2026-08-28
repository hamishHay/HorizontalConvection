import numpy as np
import dedalus.public as d3
from dedalus.core.domain import Domain
from dedalus.core.operators import UnaryGridFunction
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import time
import glob
import h5py
from diagnostics import array_diff_1D, array_mult, array_diff_2D, ice_ocean_interface_extract
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def run_horizontal_conv_sim(params):
    # Model parameters
    Lx, Lz = params['Lx'], params['Lz'] # domain size
    Tm = params['Tm'] # melt temperature
    z0 = params['z0'] # initial position of the interface 
    Ra = params['Ra'] # Rayleigh number
    Pr = params['Pr'] # Prandtl number
    S = params['S']   # Stefan number
  
    ϵ = params['ε'] # phase-field interface thickness
    γ = params['γ'] # surface tension
    
    β = params['β'] # β = 1.51044385 Optimal damping proportionality, may underestimate at large ε
    m = params['m'] # salinity induced melting temperature change
    n = params['n'] # temperature-salinity buoyancy ratio
    a = 0.0 # params['a'] # coefficient of the depth-dependent melting temperature
    b = params['b'] # bottom temperature perturbation amplitude
    δ = params['δ'] # concentration forcing regularisation
    adv = params['adv']

    file_handler_mode = 'overwrite'
    restart = params['restart']
    save_dir = params['save_dir']

    diagnostics = True

    try:
        phase = params['phase']
    except:
        phase = 1

    # Numerical parameters
    nx, nz =     params['nx'], params['nz']
    timestep =   params['timestep']
    dealias =    params['dealias']             #3/2
    stop_sim_time = params['stop_sim_time']
    snap_iter_0D =  params['snap_iter_0D']         #plot every so much time 
    snap_iter_2D =  params['snap_iter_2D']
    avg_time  =  params['avg_time']          #averaging timescale 
    chkp_time =  params['chkp_time']
    max_writes = params['max_writes']       #maximum number of plots
    print_step = params['print_step']       #terminal message written every so many time steps
    timestepper = getattr(d3,params['timestepper'])# RK222
    dtype = np.float64

    # Coordinates, Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=nz, bounds=(0, Lz), dealias=dealias)
    domain1D = Domain(dist, (xbasis,))
    domain1D_z = Domain(dist, (zbasis,))
    domain2D = Domain(dist, (xbasis, zbasis))
    domain0D = Domain(dist, bases=())
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)

    x_dealias = xbasis.global_grid(dist, scale=3/2)
    z_dealias = zbasis.global_grid(dist, scale=3/2)

    # Substitutions
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    dz = lambda A: d3.Differentiate(A,coords['z'])
    dx = lambda A: d3.Differentiate(A,coords['x'])


    # ---------------------------------------------------------------------------------
    # ----------------------------------- Fields --------------------------------------
    # ---------------------------------------------------------------------------------
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    T = dist.Field(name='T', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    xf = dist.Field(name='xf', bases=(xbasis,zbasis))
    xf['g'] = x
    zf = dist.Field(name='z', bases=(xbasis,zbasis))
    zf['g'] = z

    t = dist.Field(name="t")

    variables = [p, u, T]

    if phase:
        f = dist.Field(name='f', bases=(xbasis,zbasis))
        ft = dist.Field(name='ft', bases=(xbasis,zbasis))
        variables += [f, ft]

    # ---------------------------------------------------------------------------------
    # ------------------------- diagnostic quantities ---------------------------------
    # ---------------------------------------------------------------------------------

    xx, zz = xf['g']+0*zf['g'], 0*xf['g']+zf['g']

    xf.change_scales(dealias)
    zf.change_scales(dealias)
    xx_d, zz_d = xf['g']+0*zf['g'], 0*xf['g']+zf['g']
    xf.change_scales(1)
    zf.change_scales(1)

    diags = []


    T_bot = dist.Field(name='T_bot',bases=(xbasis))
    T_bot['g'] = 1 - b*np.cos(2*np.pi*x/Lx) # 1 here must be T_bot(z=0)

    F_bot = dist.Field(name='F_bot',bases=(xbasis))
    F_bot['g'] = -1 + b*np.cos(2*np.pi*x/Lx) # 1 here must be T_bot(z=0)

    # ---------------------------------------------------------------------------------
    # ---------------------------------- Tau terms ------------------------------------
    # ---------------------------------------------------------------------------------

    tau_p = dist.Field(name='tau_p')
    tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
    tau_T2 = dist.Field(name='tau_T2', bases=xbasis)

    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    tau_div_eq  = ez@lift(tau_u1) + tau_p
    tau_temp_eq = -dz(lift(tau_T1)) + lift(tau_T2)
    tau_mom_eq  = -dz(lift(tau_u1)) + lift(tau_u2)

    tau_terms = [tau_p, tau_T1, tau_T2, tau_u1, tau_u2]

    if phase:
        tau_f1 = dist.Field(name='tau_f1', bases=xbasis)
        tau_f2 = dist.Field(name='tau_f2', bases=xbasis)
        tau_phas_eq = -γ*dz(lift(tau_f1)) + lift(tau_f2)

        tau_terms += [tau_f1, tau_f2]


    # simpler reformulation of the tau terms
    # (d3.trace(ez*lift(tau_u1)) - ez@lift(tau_u1)).evaluate()['g'].max() # .6 faster
    # (d3.div(ez*lift(tau_T1)) - dz(lift(tau_T1))).evaluate()['g'].max() # 2.6/7.3 ~ .35 faster

    

    # ---------------------------------------------------------------------------------
    # ------------------------------ Setup problem ------------------------------------
    # ---------------------------------------------------------------------------------
    
    problem = d3.IVP(variables + tau_terms + diags, time=t, namespace=locals())

    # ---------------------------------------------------------------------------------
    # ---------------------------- Problem equations ----------------------------------
    # ---------------------------------------------------------------------------------

    problem.add_equation("div(u) + tau_div_eq = 0")

    if phase:
        problem.add_equation("dt(f) - ft = 0")
        problem.add_equation("(5/6)*S*dt(f) - γ*div(grad(f))        + tau_phas_eq = -ϵ**(-2)*f*(1-f)*(γ*(1-2*f) + (T-Tm-a*(zf-z0)))")
        problem.add_equation("dt(T) - div(grad(T)) - S*dt(f)              + tau_temp_eq = - (1-f*adv)*u@grad(T) + T*u@grad(f)*adv")
        problem.add_equation("dt(u)/Pr - div(grad(u)) + grad(p) -Ra*T*ez + tau_mom_eq  = - u@grad(u)/Pr - (1/(ϵ*β)**2)*f*u")
    else:
        problem.add_equation("dt(T) - div(grad(T))               + tau_temp_eq = - u@grad(T)")
        problem.add_equation("dt(u)/Pr - div(grad(u)) + grad(p) -Ra*T*ez + tau_mom_eq  = - u@grad(u)/Pr")

    # ---------------------------------------------------------------------------------
    # ---------------------------- Boundary conditions --------------------------------
    # ---------------------------------------------------------------------------------

    # domain top
    problem.add_equation("T(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    if phase: problem.add_equation("f(z=Lz) = 1")
    
    # domain bottom
    # problem.add_equation("T(z=0) = T_bot")
    problem.add_equation("dz(T)(z=0) = F_bot")
    problem.add_equation("u(z=0) = 0")
    if phase: problem.add_equation("f(z=0) = 0")

    problem.add_equation("integ(p) = 0") # Pressure gauge

    
    # ---------------------------------------------------------------------------------
    # ------------------------------- Build solver ------------------------------------
    # ---------------------------------------------------------------------------------
    
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # ---------------------------------------------------------------------------------
    # ----------------------------- Initial conditions --------------------------------
    # ---------------------------------------------------------------------------------

    if phase: f.change_scales(1)
    u.change_scales(1)
    T.change_scales(1)

    if isinstance(restart,str):
        write, initial_timestep = solver.load_state(restart, allow_missing=True)
        file_handler_mode = 'append'
        solver.stop_sim_time += stop_sim_time
    elif restart == 0:
        u['g'] = 0
        T.fill_random('g', seed=42, distribution='normal', scale=2e-4) # Random noise
        k = 2*np.pi / Lx 
        if phase:
            mask = lambda x : 0.5*(1 + np.tanh(x/(2*ϵ)))
            f['g'] = mask(z-z0) #Initial phase field (smooth mask, liquid from 0 to z0, ice above) 
            T['g'] *= (1-np.heaviside(z-z0,1)) # Remove random fluctuations in ice
            T['g'] += np.heaviside(z-z0, 1)*Tm*(z-1)/(z0-1) # Temperature in the ice 
            # T['g'] += (1-np.heaviside(z-z0,1))*(1 + (Tm-1)*z/z0 
            #                                     - b*np.sinh(k*(z0-z)) / np.sinh(k *z0) * np.cos(k*x) ) # Temperature in the liquid
        
            T['g'] += (1-np.heaviside(z-z0,1)) *(Tm + z0 - z - b*np.sinh(k*(z0-z)) / (k*np.cosh(k *z0)) * np.cos(k*x) ) # Temperature in the liquid for heat flux condtion
        else:
            T['g'] += 1 - z - b*np.sinh(k*(1-z)) / (k*np.cosh(k)) * np.cos(k*x) 
    else:
        load_file = f'{params["save_dir"]}/chkp/chkp_s' + str(restart) + ".h5"
        write, initial_timestep = solver.load_state(load_file)
        file_handler_mode = 'append'
        # solver.stop_sim_time += stop_sim_time
        
   

    # ---------------------------------------------------------------------------------
    # -------------------------- Begin time integration -------------------------------
    # ---------------------------------------------------------------------------------

    start_time = time.time()
    try:
        while solver.proceed:
            solver.step(timestep)
                
            if solver.iteration % print_step == 0:
                log = [f'it {solver.iteration:d}',
                       f'sim time {solver.sim_time:.4f}',
                       f'wall time {(time.time() - start_time):.1f} s',
                       f'max u {np.amax(abs(u['g'])):.3f}',
                       f'isnan? {np.isnan(np.sum(u['g'])):.0f}',
                       ]
                logger.info(', '.join(log))

                if np.isnan(np.sum(u['g'])):
                    logger.error("NaN encountered. Terminating calculations.")
                    return
            
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise       
