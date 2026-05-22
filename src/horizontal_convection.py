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
    Lx, Lz = params['Lx'], params['Lz']#1, 1 # domain size
    Tm = params['Tm']#.5 melt temperature
    z0 = params['z0']#.5 initial position of the interface 
    Ra = params['Ra']#1e4 Rayleigh number
    Pr = params['Pr']#1 Prandtl number
    #Sc = params['Sc']#1 Schmidt number
    S = params['S']#1 Stefan number
    #ν = params['ν']#1e-2 # kinematic viscosity
    #κ = params['κ']#ν # thermal diffusivity
    #μ = params['μ']#ν # salt diffusivity
    ϵ = params['ε']#1e-2 # phase-field interface thickness
    γ = params['γ']#1e-2 # surface tension
    #L = params['L']#1 # Stefan number (latent heat)
    # β = 1.51044385 # Optimal damping proportionality, may underestimate at large ε
    β = params['β']
    m = params['m']#0 # salinity induced melting temperature change
    n = params['n']#1 # temperature-salinity buoyancy ratio
    a = params['a']#.05 # coefficient of the depth-dependent melting temperature
    b = params['b']#.05 # bottom temperature perturbation amplitude
    δ = params['δ']#1e-2 # concentration forcing regularisation
    adv = params['adv']

    #Dimensionless parameters
    #Pr = ν/κ #Prandtl 
    #Sc = ν/μ #Schmidt
    #Ra = 1.0*Lz**3/(κ*ν) #Rayleigh
    #S = L/1.0 #Stefan L/deltaT=Lambda/(cp*deltaT)
    G = γ/ϵ #γ/(ϵ*deltaT)  
    E = (Lz/ϵ)**2 


    # Numerical parameters
    nx, nz = params['nx'], params['nz']#256, 256
    timestep = params['timestep']
    dealias = params['dealias']#3/2
    stop_sim_time = params['stop_sim_time']
    save_time = params['save_time'] #plot every so much time 
    max_writes = params['max_writes'] #maximum number of plots
    print_step = params['print_step'] #terminal message written every so many time steps
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
    #C = dist.Field(name='C', bases=(xbasis,zbasis))
    f = dist.Field(name='f', bases=(xbasis,zbasis))
    ft = dist.Field(name='ft', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    zf = dist.Field(name='z', bases=(xbasis,zbasis))
    zf['g'] = z
    T_bot = dist.Field(name='T_bot',bases=(xbasis))
    T_bot['g'] = 1 - b*np.cos(2*np.pi*x/Lx) #+ b*np.cos((2*np.pi/Lx)*(x-Lx/2)) # 1 here must be T_bot(z=0)
    # T_bot['g'] = 1 + 0.5*np.cos((4*np.pi/Lx)*(x-Lx/2))

    tau_p = dist.Field(name='tau_p')
    tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
    tau_T2 = dist.Field(name='tau_T2', bases=xbasis)
    #tau_C1 = dist.Field(name='tau_C1', bases=xbasis)
    #tau_C2 = dist.Field(name='tau_C2', bases=xbasis)
    tau_f1 = dist.Field(name='tau_f1', bases=xbasis)
    tau_f2 = dist.Field(name='tau_f2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    dz = lambda A: d3.Differentiate(A,coords['z'])
    dx = lambda A: d3.Differentiate(A,coords['x'])

    tau_div_eq = ez@lift(tau_u1) + tau_p
    tau_temp_eq = -dz(lift(tau_T1)) + lift(tau_T2)
    #tau_conc_eq = -Pr/Sc*dz(lift(tau_C1)) + lift(tau_C2)
    tau_phas_eq = -G*dz(lift(tau_f1)) + lift(tau_f2)
    tau_mom_eq  = -dz(lift(tau_u1)) + lift(tau_u2)

    # # Volume penalty walls
    mask = lambda x : 0.5*(1 + np.tanh(x/(2*ϵ)))
    # wall = dist.Field(name='wall', bases=(xbasis,zbasis))
    # wall['g'] = mask(-(x-Lx/20)) + mask(x-Lx*(1-1/20))
    wall = 0

    # integral quantities
    momentum = d3.integ(u)
    heat = d3.integ(T) - S*d3.integ(f)
    vorticity = -d3.div(d3.skew(u))

    #salt = d3.integ((1-f)*C)

    # boundary quantities
    heat_flux = dz(T)(z=Lz) - dz(T)(z=0)

    # # simpler reformulation of the tau terms
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
    #problem.add_equation("dt(C) - Pr/Sc*div(grad(C))                        + tau_conc_eq = - u@grad(C) + (C*ft - Pr/Sc*grad(f)@grad(C))/(1-f+δ)")# + (1-f)*μ*grad(wall)@grad(C)/(1-wall+δ)")
    problem.add_equation("(5/6)*S*dt(f) - G*div(grad(f))        + tau_phas_eq = -E*f*(1-f)*(G*(1-2*f) + (T-Tm-a*(zf-z0)))")
    problem.add_equation("dt(u)/Pr - div(grad(u)) + grad(p) -Ra*T*ez + tau_mom_eq  = - u@grad(u)/Pr - (E/β**2)*(f+wall)*u")

    problem.add_equation("T(z=Lz) = 0")
    #problem.add_equation("dz(C)(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("f(z=Lz) = 1")
    problem.add_equation("integ(p) = 0") # Pressure gauge
    problem.add_equation("T(z=0) = T_bot")
    #problem.add_equation("dz(C)(z=0) = 0")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("f(z=0) = 0")

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    f.change_scales(1)
    f['g'] = mask(z-z0) #Initial phase field (smooth mask, liquid from 0 to z0, ice above) 
    xx, zz = x+0*z, 0*x+z

    u.change_scales(1)
    u['g'] = 0
    # u['g'] = (ex['g']*(1-f['g'])*z*(Lz/2-z))
    # u['g'][0][zz>Lz/2] = 0

    # Initial conditions
    T.change_scales(1)
    T.fill_random('g', seed=42, distribution='normal', scale=2e-4) # Random noise
    # T['g'] += 1 - 2*z/Lz
    # T['g'] = (1-f['g'])*(1-z/Lz)
    T['g'] += np.heaviside(z-z0,1)*Tm*(z-1)/(z0-1) + (1-np.heaviside(z-z0,1))*(1+(Tm-1)*z/z0) #first term: in solid, second:in liquid
    #T['g'] += b*np.cos((2*np.pi/Lx)*(x-Lx/2))*(1-f['g'])

    # Analysis
    snapshots = solver.evaluator.add_file_handler(f'data/snapshots-{params["sim_name"]}', sim_dt=save_time, max_writes=max_writes)
    snapshots.add_tasks(solver.state)
    snapshots.add_task(momentum,name='momentum')
    snapshots.add_task(heat,name='heat')
    #snapshots.add_task(salt,name='salt')
    snapshots.add_task(heat_flux,name='heat_flux')
    snapshots.add_task(vorticity, name='vorticity')

    diagnostics = solver.evaluator.add_file_handler(f'data/diagnostics-global-{params["sim_name"]}', iter=100)
    # diagnostics.add_task(f*u@u, name='KE solid')
    # diagnostics.add_task((1-f)*u@u, name='KE liquid')
    diagnostics.add_task(d3.Integrate(f*u@u,     ('x', 'z')), name='KE solid global')
    diagnostics.add_task(d3.Integrate((1-f)*u@u, ('x', 'z')), name='KE liquid global')
    diagnostics.add_task(d3.Integrate(f, ('x', 'z')), name='f total')

    # # CFL
    # CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
    #              max_change=1.5, min_change=0.5, max_dt=max_timestep)
    # CFL.add_velocity(u)

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
                       # f'max c {flow.max("conc"):.3f}',
                       # f'max h {flow.max("depth"):.3f}',
                       # f'heat {heat.evaluate()["g"][0,0]:.3f}',
                       # f'salt {salt.evaluate()["g"][0,0]:.3f}',
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

