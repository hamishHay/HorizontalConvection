import numpy as np
from scipy.interpolate import interp1d
import dedalus.public as d3
import h5py 
import glob


def compute_streamfunction(u, coords, xbasis, zbasis, dist, check=True, tol=1e-6):
    """
    Streamfunction psi with u@ex = dz(psi), u@ez = -dx(psi), via vertical
    integration of the x-velocity component. Valid for a divergence-free
    VectorField (e.g. an equilibrated no-ice DNS state). Assumes z=0 is
    the impermeable bottom.
    """
    ex, ez = coords.unit_vector_fields(dist)
    dz = lambda A: d3.Differentiate(A, coords['z'])
    dx = lambda A: d3.Differentiate(A, coords['x'])
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    psi = dist.Field(name='psi', bases=(xbasis, zbasis))
    tau_psi = dist.Field(name='tau_psi', bases=xbasis)

    problem = d3.LBVP([psi, tau_psi], namespace=locals())
    problem.add_equation("dz(psi) + lift(tau_psi) = u@ex")
    problem.add_equation("psi(z=0) = 0")
    solver = problem.build_solver()
    solver.solve()

    if check:
        w_check = (-dx(psi)).evaluate()
        w_check.change_scales(1)
        uz = (u @ ez).evaluate()
        uz.change_scales(1)
        resid = np.max(np.abs(w_check['g'] - uz['g']))
        rel = resid / (np.max(np.abs(uz['g'])) + 1e-30)
        print(f"[compute_streamfunction] relative residual = {rel:.3e}")
        if rel > tol:
            print(f"  WARNING: residual exceeds tol={tol:.1e} — "
                  f"input u may not be sufficiently divergence-free.")

    return psi

def ice_thickness_from_flux(F_x, Lz, Tm, T_top=0.0, h_min=0.02, h_max=None,
                             smooth_modes=None, dist=None, xbasis=None):
    """
    Convert an upward heat flux profile F(x) into an ice thickness via
    the local quasi-steady Stefan balance (k_ice/k_liquid = 1):

        h(x) ~ (Tm - T_top) / F(x)

    F_x : 1D numpy array of (positive) upward flux, already time- and/or
          horizontally-resolved as you want it (e.g. loaded from the
          'heat flux top avg x' diagnostic of a no-ice run).
    smooth_modes : optional low-pass in x (needs dist, xbasis to build a
          scratch Field for the Fourier transform).
    """
    F_x = np.maximum(np.abs(F_x), 1e-8)

    Fmean = np.mean(F_x)
    Fpert = F_x - Fmean
    Fpert *= 0.85   # Apply scaling factor to reduce ice thickness variations
                    # which seems to help with 2D effects
    F_x = Fmean + Fpert

    B = 1 / F_x 
    Bmean = np.mean(B)
    Bpert = B - Bmean 

    Hmean = (Tm - T_top) * Bmean
    Hpert = (Tm - T_top) * Bpert 

    h_x = Hmean + Hpert

    print("Average ice thickness is", Hmean)

    if h_max is None:
        h_max = 0.9 * Lz
    h_x = np.clip(h_x, h_min, h_max)

    if smooth_modes is not None:
        assert dist is not None and xbasis is not None
        h_field = dist.Field(name='h_smooth', bases=xbasis)
        h_field.change_scales(1)
        h_field['g'][:,0] = h_x
        h_field['c'][smooth_modes:] = 0
        h_field.change_scales(1)
        h_x = np.clip(h_field['g'].copy(), h_min, h_max)

    z0_x = Lz - h_x

    return h_x, z0_x

def load_time_avg_heat_flux_top(diags_dir, n_last=50):
    """
    Read the 'heat flux top avg x' task from the no-ice run's tavg_integ
    file handler (params['save_dir']/diags/*.h5) and return the profile
    from the last write (or averaged over the last n_last writes, which
    further smooths over separate averaging windows).

    Note: each write of 'heat flux top avg x' is itself already a
    cumulative time-integral divided by int_time (see how it's added in
    the sim: dt(avg_dTdz_t_x) = heat_flux_top, normalized by int_time at
    write time) — i.e. already a proper time average over that window,
    not a snapshot. Taking the LAST write gives you the average over the
    final (presumably best-equilibrated) window; averaging the last few
    writes further reduces window-to-window variability.
    """
    files = sorted(glob.glob(f'{diags_dir}/diags_s*.h5'))
    if not files:
        raise FileNotFoundError(f'No diags files found in {diags_dir}')

    fname = files[-1]
    F_x = 0
    with h5py.File(fname, 'r') as h5f:
        key = [k for k in h5f['tasks'].keys()
                if 'heat flux top avg x' in k][0]
        data = h5f['tasks'][key][()]   # shape (n_writes, nx, 1) typically
        F_x = np.mean(data[-n_last:, :, 0], axis=0)


    print("Mean flux is ", np.mean(F_x))
    return F_x

def remap_fields_to_ice_thickness(psi_old, T_old, z0_x, coords, dist,
                                   xbasis, zbasis_old, zbasis_new,
                                   Lz_old, Lz_new, Tm, T_ice_top=0.0):
    """
    Squeeze the old (no-ice) fields — living on domain height Lz_old —
    into the new liquid layer 0 <= z <= z0_x(x) of the phase-field
    domain (height Lz_new = Lz_old + Tm), by resampling each x-column at
    rescaled z. Differentiating the remapped psi (on zbasis_new)
    spectrally then gives a divergence-free (u, w) by construction.
    """
    psi_old.change_scales(1)
    T_old.change_scales(1)
    psi_g = psi_old['g'].copy()   # (nx, nz_old)
    T_g = T_old['g'].copy()

    _, z_old = dist.local_grids(xbasis, zbasis_old)
    z_old_col = z_old.ravel()

    x_new, z_new = dist.local_grids(xbasis, zbasis_new)
    z_new_col = z_new.ravel()

    nx_local = psi_g.shape[0]
    nz_new = z_new_col.size

    psi_new = np.zeros((nx_local, nz_new))
    T_new = np.zeros((nx_local, nz_new))
    T_ice = np.zeros((nx_local, nz_new))

    for i in range(nx_local):
        h_i = float(z0_x[i, 0]) if np.ndim(z0_x) == 2 else float(z0_x[i])
        h_i = max(h_i, 1e-6)

        # sample the OLD [0, Lz_old] column at coords rescaled into
        # the NEW liquid layer [0, h_i]
        z_sample = np.clip(z_new_col * (Lz_old / h_i), 0, Lz_old)

        f_psi = interp1d(z_old_col, psi_g[i, :], kind='cubic',
                          bounds_error=False, fill_value='extrapolate')
        f_T = interp1d(z_old_col, T_g[i, :], kind='cubic',
                        bounds_error=False, fill_value='extrapolate')

        psi_new[i, :] = f_psi(z_sample)
        T_new[i, :] = f_T(z_sample)

        # conductive-linear ice temperature, Tm at interface -> T_ice_top at Lz_new
        T_ice[i, :] = np.where(z_new_col > h_i,
                               Tm + (T_ice_top - Tm) * (z_new_col - h_i) / (Lz_new - h_i),
                               0.0)

    psi_field = dist.Field(name='psi_remapped', bases=(xbasis, zbasis_new))
    psi_field.change_scales(1)
    psi_field['g'] = psi_new

    dz = lambda A: d3.Differentiate(A, coords['z'])
    dx = lambda A: d3.Differentiate(A, coords['x'])
    u_x_new = dz(psi_field).evaluate()
    u_z_new = (-dx(psi_field)).evaluate()
    u_x_new.change_scales(1)
    u_z_new.change_scales(1)

    return {
        'u_x': u_x_new['g'],
        'u_z': u_z_new['g'],
        'T_liquid': T_new,
        'T_ice': T_ice,
    }


def blend_with_phase_field(z0_x, remapped, x, z, eps):
    """
    Build the tanh phase field at the new interface z0_x and blend
    velocity/temperature across it — same convention as your flat-start
    IC (mask = 0.5*(1+tanh((z-z0)/(2*eps)))).
    """

    z0_col = np.asarray(z0_x).reshape(-1, 1)  # (nx,1), broadcasts against z (1,nz)


    mask = lambda zz: 0.5 * (1 + np.tanh(zz / (2 * eps)))
    f_new = mask(z - z0_col)

    T_new = (1 - f_new) * remapped['T_liquid'] + f_new * remapped['T_ice']
    u_x_new = (1 - f_new) * remapped['u_x']
    u_z_new = (1 - f_new) * remapped['u_z']

    return f_new, T_new, u_x_new, u_z_new

def remap_piecewise_interface(z_full, psi_full, T_full, z_new_col, h_x, Lz_old, Lz):
    """
    Inverts your two-branch forward map

        liquid: z_new = z_old * (1 + h/Lz_old),                z_old in [0, Lz_old]
        ice:    z_new = z_old + h*(1 - z_old)/(1 - Lz_old),     z_old in [Lz_old, Lz]

    per x-column, to find the z_old to sample the combined old field
    at for each new-grid z_new point. h=0 gives the identity map.
    """
    nx_local = psi_full.shape[0]
    nz_new = z_new_col.size
    psi_new = np.zeros((nx_local, nz_new))
    T_new = np.zeros((nx_local, nz_new))

    for i in range(nx_local):
        h_i = float(h_x[i, 0]) if np.ndim(h_x) == 2 else float(h_x[i])
        z0_i = Lz_old + h_i

        z_old_sample = np.empty(nz_new)
        below = z_new_col <= z0_i
        above = ~below

        z_old_sample[below] = z_new_col[below] / (1 + h_i / Lz_old)
        denom = 1 - h_i / (1 - Lz_old)
        z_old_sample[above] = (z_new_col[above] - h_i / (1 - Lz_old)) / denom
        z_old_sample = np.clip(z_old_sample, 0, Lz)

        f_psi = interp1d(z_full, psi_full[i, :], kind='cubic',
                          bounds_error=False, fill_value='extrapolate')
        f_T = interp1d(z_full, T_full[i, :], kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
        psi_new[i, :] = f_psi(z_old_sample)
        T_new[i, :] = f_T(z_old_sample)

    return psi_new, T_new

def load_mapped_initial_condition(filepath, dist, coords, xbasis, zbasis, x, z):
    """
    Read a global psi/T/z0_x reference-grid .h5 file (written serially
    by build_initial_condition) and interpolate onto this process's
    local (x, z) grid — works for any core count / resolution mismatch.
    psi is differentiated AFTER interpolation, using this run's own
    Dedalus operators, so u=dz(psi), w=-dx(psi) is divergence-free on
    the actual local discretization rather than inherited approximately
    from the reference grid.
    """
    with h5py.File(filepath, 'r') as h5f:
        psi_ref = h5f['psi'][()]     # (nx_ref, nz_ref) — global, every rank reads this
        T_ref = h5f['T'][()]
        z0_ref = h5f['z0_x'][()]     # (nx_ref,)
        x_ref = h5f['x_ref'][()]
        z_ref = h5f['z_ref'][()]
        Lx = h5f.attrs['Lx']

    x_local = x.ravel()
    z_local = z.ravel()

    def interp_x_periodic(arr_1d, x_ref, x_local, Lx, pad=4):
        # wrap-pad both ends before cubic interp so periodicity at x=0/Lx
        # is respected, rather than extrapolating past the edge
        dx_ref = x_ref[1] - x_ref[0]
        x_pad = np.concatenate([x_ref[-pad:] - Lx, x_ref, x_ref[:pad] + Lx])
        a_pad = np.concatenate([arr_1d[-pad:], arr_1d, arr_1d[:pad]])
        f = interp1d(x_pad, a_pad, kind='cubic')
        return f(np.mod(x_local, Lx))

    def interp2d_field(field_ref):
        # stage 1: interpolate along x (periodic) for every reference z row
        nz_ref = field_ref.shape[1]
        stage1 = np.zeros((x_local.size, nz_ref))
        for k in range(nz_ref):
            stage1[:, k] = interp_x_periodic(field_ref[:, k], x_ref, x_local, Lx)

        # stage 2: interpolate along z (non-periodic, Chebyshev) per local column
        out = np.zeros((x_local.size, z_local.size))
        for i in range(x_local.size):
            f = interp1d(z_ref, stage1[i, :], kind='cubic',
                         bounds_error=False, fill_value='extrapolate')
            out[i, :] = f(z_local)
        return out

    psi_local_arr = interp2d_field(psi_ref)
    T_local_arr = interp2d_field(T_ref)
    z0_local = interp_x_periodic(z0_ref, x_ref, x_local, Lx)

    # build psi on THIS run's actual (possibly distributed) bases, then
    # differentiate with the run's own operators — divergence-free by
    # construction on the local grid, not just approximately preserved
    # through interpolation
    psi_field = dist.Field(name='psi_local', bases=(xbasis, zbasis))
    psi_field.change_scales(1)
    psi_field['g'] = psi_local_arr

    dz = lambda A: d3.Differentiate(A, coords['z'])
    dx = lambda A: d3.Differentiate(A, coords['x'])
    u_x_field = dz(psi_field).evaluate()
    u_z_field = (-dx(psi_field)).evaluate()
    u_x_field.change_scales(1)
    u_z_field.change_scales(1)

    return u_x_field['g'], u_z_field['g'], T_local_arr, z0_local