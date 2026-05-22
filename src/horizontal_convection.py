import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import time
import glob
import h5py

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
    save_time =  params['save_time']         #plot every so much time 
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
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)

    # Fields
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    T = dist.Field(name='T', bases=(xbasis,zbasis))
    f = dist.Field(name='f', bases=(xbasis,zbasis))
    ft = dist.Field(name='ft', bases=(xbasis,zbasis))
    
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
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

    # Substitutions
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    dz = lambda A: d3.Differentiate(A,coords['z'])
    dx = lambda A: d3.Differentiate(A,coords['x'])

    tau_div_eq  = ez@lift(tau_u1) + tau_p
    tau_temp_eq = -dz(lift(tau_T1)) + lift(tau_T2)
    tau_phas_eq = -γ*dz(lift(tau_f1)) + lift(tau_f2)
    tau_mom_eq  = -dz(lift(tau_u1)) + lift(tau_u2)

    # integral quantities
    momentum = d3.integ(u)
    heat = d3.integ(T) - S*d3.integ(f)
    vorticity = -d3.div(d3.skew(u))
    
    Re = np.sqrt(d3.Average(u@u ('x', 'y'))
    Nu_RB = d3.Average(dz(T)(z=Lz), ('x'))


    # boundary quantities
    heat_flux_top = dz(T)(z=Lz) 
    heat_flux_bot = dz(T)(z=0)

    # simpler reformulation of the tau terms
    # (d3.trace(ez*lift(tau_u1)) - ez@lift(tau_u1)).evaluate()['g'].max() # .6 faster
    # (d3.div(ez*lift(tau_T1)) - dz(lift(tau_T1))).evaluate()['g'].max() # 2.6/7.3 ~ .35 faster

    # Problem
    problem = d3.IVP([p, u, T, f, ft,  
                      tau_p, tau_T1, tau_T2, tau_f1, tau_f2, 
                      tau_u1, tau_u2, 
                      ], namespace=locals())

    problem.add_equation("dt(f) - ft = 0")
    problem.add_equation("div(u) + tau_div_eq = 0")
    problem.add_equation("dt(T) - div(grad(T)) - S*dt(f)              + tau_temp_eq = - (1-f*adv)*u@grad(T) + T*u@grad(f)*adv")
    problem.add_equation("(5/6)*S*dt(f) - γ*div(grad(f))        + tau_phas_eq = -ϵ**(-2)*f*(1-f)*(γ*(1-2*f) + (T-Tm-a*(zf-z0)))")
    problem.add_equation("dt(u)/Pr - div(grad(u)) + grad(p) -Ra*T*ez + tau_mom_eq  = - u@grad(u)/Pr - (1/(ϵ*β)**2)*f*u")

    # Boundary conditions
    problem.add_equation("T(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("f(z=Lz) = 1")
    
    problem.add_equation("T(z=0) = T_bot")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("f(z=0) = 0")

    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    f.change_scales(1)
    u.change_scales(1)
    T.change_scales(1)
    xx, zz = x+0*z, 0*x+z

    # Initial conditions
    f['g'] = mask(z-z0) #Initial phase field (smooth mask, liquid from 0 to z0, ice above) 
    u['g'] = 0
    T.fill_random('g', seed=42, distribution='normal', scale=2e-4) # Random noise
    T['g'] += np.heaviside(z-z0,1)*Tm*(z-1)/(z0-1) + (1-np.heaviside(z-z0,1))*(1+(Tm-1)*z/z0) #first term: in solid, second:in liquid

    # checkpoints
    checkpoints = solver.evaluator.add_file_handler(f'data/chkp-{params["sim_name"]}',
                                                    sim_dt=chkp_time, max_writes=1, mode=file_handler_mode))
    checkpoints.add_tasks(solver.state)

    # Analysis
    snapshots = solver.evaluator.add_file_handler(f'data/snaps-{params["sim_name"]}', 
                                                  sim_dt=save_time, max_writes=max_writes)
    snapshots.add_task(momentum,name='momentum')
    snapshots.add_task(heat,name='heat')

    snapshots.add_task(heat_flux,name='heat_flux')
    snapshots.add_task(vorticity, name='vorticity')

    diagnostics = solver.evaluator.add_file_handler(f'data/diags-{params["sim_name"]}', iter=100)
    # diagnostics.add_task(f*u@u, name='KE solid')
    # diagnostics.add_task((1-f)*u@u, name='KE liquid')
    diagnostics.add_task(d3.Integrate(f*u@u,     ('x', 'z')), name='KE solid global')
    diagnostics.add_task(d3.Integrate((1-f)*u@u, ('x', 'z')), name='KE liquid global')
    diagnostics.add_task(d3.Integrate(f, ('x', 'z')), name='f total')

    # # Flow properties
    # flow = d3.GlobalFlowProperty(solver, cadence=10)
    # flow.add_property(np.sqrt(u@u), name='speed')
    # flow.add_property(f, name='phase')
    # flow.add_property(T, name='temp')
    # #flow.add_property(C, name='conc')
    # flow.add_property(d3.integ(f,'z'), name='depth')

    # flow.add_property(f*u@u, name='KE liquid')
    # flow.add_property((1-f)*u@u, name='KE solid')
    # flow.add_property(d3.Integrate(f*u@u, ('x', 'y')), name='KE liquid global')

    start_time = time.time()
    try:
        while solver.proceed:
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
            solver.step(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise       

import file_tools as flt

def plot_europa_sim(params):

    Lx, Lz = params['Lx'], params['Lz']#1, 1 # domain size

    data_files = sorted(glob.glob(f'data/snapshots-{params["sim_name"]}/*.h5'))
    plot_dir = f'plots/{params["sim_name"]}'
    flt.makedir(plot_dir)
    
    with h5py.File(data_files[0], 'r') as f:
        u, T, φ = [f['tasks'][name][:] for name in ['u','T','f']]
        t, x, z = [f['tasks']['u'].dims[i][n][:] for i, n in [(0,'sim_time'),(2,'x'),(3,'z')]]
    
    xx, zz = np.meshgrid(x, z, indexing='ij')

    for it in range(len(t)):
        fig, ax = plt.subplots(3,1,figsize=(3*Lx, 15))
        ps = {}
        ps[0,0] = ax[0].pcolormesh(xx, zz, T[it], cmap='RdBu_r')
        #ps[1,0] = ax[1,0].pcolormesh(xx, zz, C[it], cmap='Purples')
        ps[1,0] = ax[1].pcolormesh(xx, zz, u[it][0], cmap='RdBu_r')
        ps[2,0] = ax[2].pcolormesh(xx, zz, u[it][1], cmap='RdBu_r')
        for i in range(3):
                ax[i].contour(xx,zz,φ[it],[.05,.5,.95],colors='k', linewidths=0.5)
                plt.colorbar(ps[i,0], ax=ax[i])    

        ax[0].set_ylabel("Temperature")
        ax[1].set_ylabel("Velocity x")
        ax[2].set_ylabel("Velocity y")
        plt.savefig(f'{plot_dir}/step-{it:0>3d}.png',bbox_inches='tight')

