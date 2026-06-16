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
from diagnostics import array_diff_1D, array_diff_2D, custom_grid_function, ice_ocean_interface_extract

def run_europa_sim(params):
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

    restart = params['restart']

    file_handler_mode = 'overwrite'
    if restart > 0:
        file_handler_mode = 'append'

    # Numerical parameters
    nx, nz =     params['nx'], params['nz']
    timestep =   params['timestep']
    dealias =    params['dealias']             #3/2
    stop_sim_time = params['stop_sim_time']
    snap_time =  params['snap_time']         #plot every so much time 
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
    domain2D = Domain(dist, (xbasis, zbasis))
    domain0D = Domain(dist, bases=())
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)

    # Substitutions
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    dz = lambda A: d3.Differentiate(A,coords['z'])
    dx = lambda A: d3.Differentiate(A,coords['x'])

    def S1(*args, domain=domain1D, F=array_diff_1D, ten=()):
        return d3.GeneralFunction(
            dist=dist,
            domain=domain,
            tensorsig=ten,
            dtype=np.float64,
            layout="g",
            func=F,
            args=args,
        )
    
    def S2(*args, domain=domain2D, F=array_diff_1D):
        return d3.GeneralFunction(
            dist=dist,
            domain=domain,
            tensorsig=(coords,),
            dtype=np.float64,
            layout="g",
            func=F,
            args=args,
        )
    

    # ---------------------------------------------------------------------------------
    # ----------------------------------- Fields --------------------------------------
    # ---------------------------------------------------------------------------------
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    T = dist.Field(name='T', bases=(xbasis,zbasis))
    f = dist.Field(name='f', bases=(xbasis,zbasis))
    ft = dist.Field(name='ft', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))

    t = dist.Field(name="t")

    # ---------------------------------------------------------------------------------
    # ------------------------- diagnostic quantities ---------------------------------
    # ---------------------------------------------------------------------------------

    V_liq = d3.integ(1-f)
    V_ice = d3.integ(f) # Volume of ice = total volume - volume of liquid

    KE = d3.Average(u@u)
    KE_ice = d3.integ(u@u * f) / V_ice
    KE_liq = d3.integ(u@u * (1-f)) / V_liq

    # heat = d3.integ(T) - S*d3.integ(f)
    vorticity = -d3.div(d3.skew(u))

    Re_liq = d3.integ(np.sqrt(u@u) * (1-f), ('x', 'z')) / V_liq
    Nu_RB = d3.Average(dz(T)(z=Lz), ('x'))
    f_profile = d3.Average(f, ('z'))
    # Nu_RB = d3.Average(dz(T)(z=Lz), ('x'))

    # boundary quantities
    heat_flux_top = dz(T)(z=Lz) 
    heat_flux_bot = dz(T)(z=0)

    # ---------------------------------------------------------------------------------
    # ------------------------ time-average diagnostics -------------------------------
    # ---------------------------------------------------------------------------------


    avg_KE = dist.Field(name='avg_KE')                      # total kinetic energy
    avg_KE0 = np.zeros((1,1))

    avg_KE_ice = dist.Field(name='avg_KE_ice')              # total kinetic energy
    avg_KE_ice0 = np.zeros((1,1))

    avg_KE_liq = dist.Field(name='avg_KE_liq')              # total kinetic energy
    avg_KE_liq0 = np.zeros((1,1))

    avg_f_x = dist.Field(name='avg_f_x', bases=(xbasis))    # phase-topology profile
    avg_f_x0 = np.zeros((3*nx // 2, 1))

    avg_dTdz_t_x = dist.Field(name='avg_dTdz_t_x', bases=(xbasis))  # heat flux top
    avg_dTdz_t_x0 = np.zeros((3*nx // 2, 1))

    avg_dTdz_b_x = dist.Field(name='avg_dTdz_b_x', bases=(xbasis))  # heat flux bot
    avg_dTdz_b_x0 = np.zeros((3*nx // 2, 1))

    avg_Nu = dist.Field(name='Nu')          # Nusselt number at the top 
    avg_Nu0 = np.zeros((1,1))

    # avg_Nu_io = dist.Field(name='Nu_io')    # Nusselt number at the ocean--ocean interface
    # avg_Nu_io0 = 0.0

    avg_Re = dist.Field(name="Re")          # Reynolds number in the liquid
    avg_Re0 = np.zeros((1,1))

    avg_u = dist.VectorField(coords, name='avg_u', bases=(xbasis,zbasis), dtype=dtype)
    avg_u0 = np.zeros((2, 3*nx // 2, 3*nz // 2), dtype=dtype)

    avg_T = dist.Field(name="avg_T", bases=(xbasis,zbasis))
    avg_T0 = np.zeros((3*nx // 2, 3*nz // 2))

    avg_f = dist.Field(name="avg_f", bases=(xbasis,zbasis))
    avg_f0 = np.zeros((3*nx // 2, 3*nz // 2))


    diags = [avg_KE, avg_f_x, avg_dTdz_b_x, avg_dTdz_t_x, avg_Nu, avg_Re, avg_KE_ice, avg_KE_liq, avg_u, avg_T, avg_f]

    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------

    # integral quantities
    


    zf = dist.Field(name='z', bases=(xbasis,zbasis))
    zf['g'] = z

    T_bot = dist.Field(name='T_bot',bases=(xbasis))
    T_bot['g'] = 1 - b*np.cos(2*np.pi*x/Lx) # 1 here must be T_bot(z=0)

    # Tau terms
    tau_p = dist.Field(name='tau_p')
    tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
    tau_T2 = dist.Field(name='tau_T2', bases=xbasis)

    tau_f1 = dist.Field(name='tau_f1', bases=xbasis)
    tau_f2 = dist.Field(name='tau_f2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    

    tau_div_eq  = ez@lift(tau_u1) + tau_p
    tau_temp_eq = -dz(lift(tau_T1)) + lift(tau_T2)
    tau_phas_eq = -γ*dz(lift(tau_f1)) + lift(tau_f2)
    tau_mom_eq  = -dz(lift(tau_u1)) + lift(tau_u2)

    # simpler reformulation of the tau terms
    # (d3.trace(ez*lift(tau_u1)) - ez@lift(tau_u1)).evaluate()['g'].max() # .6 faster
    # (d3.div(ez*lift(tau_T1)) - dz(lift(tau_T1))).evaluate()['g'].max() # 2.6/7.3 ~ .35 faster

    # Problem
    variables = [p, u, T, f, ft,  
                      tau_p, tau_T1, tau_T2, tau_f1, tau_f2, 
                      tau_u1, tau_u2]
    problem = d3.IVP(variables + diags, time=t, namespace=locals())

    # ---------------------------------------------------------------------------------
    # ---------------------------- Problem equations ----------------------------------
    # ---------------------------------------------------------------------------------

    problem.add_equation("dt(f) - ft = 0")
    problem.add_equation("div(u) + tau_div_eq = 0")
    problem.add_equation("dt(T) - div(grad(T)) - S*dt(f)              + tau_temp_eq = - (1-f*adv)*u@grad(T) + T*u@grad(f)*adv")
    problem.add_equation("(5/6)*S*dt(f) - γ*div(grad(f))        + tau_phas_eq = -ϵ**(-2)*f*(1-f)*(γ*(1-2*f) + (T-Tm-a*(zf-z0)))")
    problem.add_equation("dt(u)/Pr - div(grad(u)) + grad(p) -Ra*T*ez + tau_mom_eq  = - u@grad(u)/Pr - (1/(ϵ*β)**2)*f*u")

    # ---------------------------------------------------------------------------------
    # ---------------------------- Boundary conditions --------------------------------
    # ---------------------------------------------------------------------------------

    # domain top
    problem.add_equation("T(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("f(z=Lz) = 1")
    
    # domain bottom
    problem.add_equation("T(z=0) = T_bot")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("f(z=0) = 0")

    problem.add_equation("integ(p) = 0") # Pressure gauge

    # ---------------------------------------------------------------------------------
    # -------------------------- Time-averaging equations -----------------------------
    # ---------------------------------------------------------------------------------

    problem.add_equation("dt(avg_KE)       = KE")
    problem.add_equation("dt(avg_KE_ice)   = KE_ice")
    problem.add_equation("dt(avg_KE_liq)   = KE_liq")
    problem.add_equation("dt(avg_f_x)      = f_profile")
    problem.add_equation("dt(avg_dTdz_t_x) = heat_flux_top")
    problem.add_equation("dt(avg_dTdz_b_x) = heat_flux_bot")

    problem.add_equation("dt(avg_Nu) = Nu_RB")
    problem.add_equation("dt(avg_Re) = Re_liq")

    problem.add_equation("dt(avg_u) - u = 0")
    problem.add_equation("dt(avg_T) - T = 0")
    problem.add_equation("dt(avg_f) - f = 0")

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    f.change_scales(1)
    u.change_scales(1)
    T.change_scales(1)
    avg_f_x.change_scales(1)
    xx, zz = x+0*z, 0*x+z

    # Masks for centre--edges diagnostics
    mask_center = (xx >= Lx/4 ) & (xx <= 3*Lx/4) 
    mask_edges =  (xx < Lx/4) | (xx > 3*Lx/4) 
    mask_bot_left = (xx < Lx/2) & (zz < 0.5)
    mask_top_left = (xx < Lx/2) & (zz >= 0.5)

    # Initial conditions
    mask = lambda x : 0.5*(1 + np.tanh(x/(2*ϵ)))
    f['g'] = mask(z-z0) #Initial phase field (smooth mask, liquid from 0 to z0, ice above) 
    u['g'] = 0
    T.fill_random('g', seed=42, distribution='normal', scale=2e-4) # Random noise
    T['g'] += np.heaviside(z-z0,1)*Tm*(z-1)/(z0-1) + (1-np.heaviside(z-z0,1))*(1+(Tm-1)*z/z0) #first term: in solid, second:in liquid
    avg_f_x['g'] = 0.0
    
    # checkpoints
    checkpoints = solver.evaluator.add_file_handler(f'data/chkp-{params["sim_name"]}',
                                                    sim_dt=chkp_time, max_writes=1, mode=file_handler_mode)
    checkpoints.add_tasks(solver.state)

    # Analysis
    # 2D snapshots 
    snapshots = solver.evaluator.add_file_handler(f'data/snaps2D-{params["sim_name"]}', 
                                                  iter=snap_time, max_writes=max_writes)

    snapshots.add_task(vorticity, name='vorticity')
    snapshots.add_task(u, name='velocity')
    snapshots.add_task(f, name='phase')
    snapshots.add_task(T, name='temperature')

    # 0D and 1D snapshots of space-integral quantities
    snapshots_integ = solver.evaluator.add_file_handler(f'data/snaps-{params["sim_name"]}', 
                                                  iter=snap_time, max_writes=max_writes)

    snapshots_integ.add_task(heat_flux_top, name='heat flux top x')
    snapshots_integ.add_task(heat_flux_bot, name='heat flux bot x')
    snapshots_integ.add_task(f_profile,     name='f x')
    snapshots_integ.add_task(V_ice,         name='vol ice')
    snapshots_integ.add_task(V_liq,         name='vol liq')
    snapshots_integ.add_task(KE_ice,        name='KE ice')
    snapshots_integ.add_task(KE,            name='KE')
    snapshots_integ.add_task(KE_liq,        name='KE liq')
    snapshots_integ.add_task(Nu_RB,         name='Nu')

    int_time = (avg_time-1) * timestep # The -1 is important! 

    tavg = solver.evaluator.add_file_handler(f'data/diags2D-{params["sim_name"]}', iter=avg_time)
    tavg.add_task(S1(avg_u, avg_u0, domain=domain2D, ten=(coords,))/int_time, name='velocity avg')
    tavg.add_task(S1(avg_T, avg_T0, domain=domain2D)/int_time, name='temperature avg')
    tavg.add_task(S1(avg_f, avg_f0, domain=domain2D)/int_time, name='phase avg')

    tavg_integ = solver.evaluator.add_file_handler(f'data/diags-{params["sim_name"]}', iter=avg_time)
    tavg_integ.add_task(S1(avg_KE,     avg_KE0,     domain=domain0D)/int_time, name='KE avg')
    tavg_integ.add_task(S1(avg_KE_ice, avg_KE_ice0, domain=domain0D)/int_time, name='KE avg ice')
    tavg_integ.add_task(S1(avg_KE_liq, avg_KE_liq0, domain=domain0D)/int_time, name='KE avg liq')
    tavg_integ.add_task(S1(avg_Nu,     avg_Nu0,     domain=domain0D)/int_time, name='Nu avg')
    tavg_integ.add_task(S1(avg_Re,     avg_Re0,     domain=domain0D)/int_time, name='Re avg')

    tavg_integ.add_task(S1(avg_f_x,      avg_f_x0)/int_time, name='f avg x')
    tavg_integ.add_task(S1(avg_dTdz_t_x, avg_dTdz_t_x0)/int_time, name='heat flux top avg x')
    tavg_integ.add_task(S1(avg_dTdz_b_x, avg_dTdz_b_x0)/int_time, name='heat flux bot avg x')
    # diagnostics.add_task(S1(dz(T), f, F=ice_ocean_interface_extract)/int_time, name='dTdz io x')

    start_time = time.time()
    try:
        while solver.proceed:
            solver.step(timestep)
            if (solver.iteration-1) % avg_time == 0:   
                # Reset variables for averaging 
                avg_KE0[:,:] = avg_KE['g']
                avg_KE_ice0[:,:] = avg_KE_ice['g']
                avg_KE_liq0[:,:] = avg_KE_liq['g']

                avg_Nu0[:,:] = avg_Nu['g']
                avg_Re0[:,:] = avg_Re['g']
                
                avg_f_x.change_scales(3/2)
                avg_f_x0[:] = avg_f_x['g']

                avg_dTdz_t_x.change_scales(3/2)
                avg_dTdz_t_x0[:] = avg_dTdz_t_x['g']

                avg_dTdz_b_x.change_scales(3/2)
                avg_dTdz_b_x0[:] = avg_dTdz_b_x['g']

                avg_u.change_scales(3/2)
                avg_u0[:, :, :] = avg_u['g']

                avg_T.change_scales(3/2)
                # print(avg_T0[:,:])
                avg_T0[:,:] = avg_T['g']

                avg_f.change_scales(3/2)
                avg_f0[:,:] = avg_f['g'][:,:]

                # KE_ice0 = KE_ice
                # KE_liq0 = KE_liq 
                # KE0     = KE


            if solver.iteration % print_step == 0:
                log = [f'it {solver.iteration:d}',
                       f'sim time {solver.sim_time:.2f}',
                       f'wall time {(time.time() - start_time):.1f} s',
                       f'max u {np.amax(abs(u['g'])):.3f}',
                       f'isnan? {np.isnan(np.sum(u['g'])):.3f}',
                       ]
                logger.info(', '.join(log))

                if np.isnan(np.sum(u['g'])):
                    logger.error("NaN encountered. Terminating calculations.")
                    return
            
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise       
