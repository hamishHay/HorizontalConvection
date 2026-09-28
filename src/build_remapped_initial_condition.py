"""
Offline script: build a remapped (u, T) initial condition for a
phase-field ice run from an equilibrated no-ice checkpoint plus its
time-averaged top heat flux. Run once, separately from the phase-field
simulation itself.

Writes initial_condition_mapped.h5 containing:
  u      : (2, nx, nz)  velocity, already stretched + masked to (1-f)
  T      : (nx, nz)     temperature, blended across the ice interface
  z0_x   : (nx,)        interface height used to build the above
           (needed by the phase-field script to reconstruct f
           consistently — see note below)
"""
import numpy as np
import h5py
import dedalus.public as d3
import matplotlib.pyplot as plt

from remap import (compute_streamfunction, ice_thickness_from_flux,
                    load_time_avg_heat_flux_top,
                    remap_piecewise_interface)

def build_old_domain_fields(u, T, coords, xbasis, zbasis_old, dist,
                             Lz_old, Lz, Tm, T_ice_top=0.0, nz_ice=None):
    """
    Extend the no-ice run's (psi, T) — defined on z in [0, Lz_old] — up
    to the full new-domain height [0, Lz] with a placeholder "ice"
    region. psi is constant above Lz_old (u=0 there, matching the
    ice-side no-flow condition); T follows the same conductive-linear
    profile used in your flat-start IC.
    """
    psi_old = compute_streamfunction(u, coords, xbasis, zbasis_old, dist)
    psi_old.change_scales(1)
    T.change_scales(1)
    psi_liquid = psi_old['g'].copy()      # (nx, nz_old)
    T_liquid = T['g'].copy()

    x, z_old = dist.local_grids(xbasis, zbasis_old)
    z_old_col = z_old.ravel()
    nz_old = z_old_col.size
    nz_ice = nz_old if nz_ice is None else nz_ice

    z_ice_col = np.linspace(Lz_old, Lz, nz_ice + 1)[1:]  # matches your example

    nx_local = psi_liquid.shape[0]
    psi_ice = np.repeat(psi_liquid[:, -1:], nz_ice, axis=1)
    T_ice = Tm + (T_ice_top - Tm) * (z_ice_col[None, :] - Lz_old) / (Lz - Lz_old)
    T_ice = np.broadcast_to(T_ice, (nx_local, nz_ice)).copy()

    psi_full = np.concatenate([psi_liquid, psi_ice], axis=1)
    T_full = np.concatenate([T_liquid, T_ice], axis=1)
    z_full = np.concatenate([z_old_col, z_ice_col])

    return x, z_full, psi_full, T_full

def differentiate_psi(psi_new_array, coords, xbasis, zbasis_new, dist):
    psi_field = dist.Field(name='psi_remapped', bases=(xbasis, zbasis_new))
    psi_field.change_scales(1)
    psi_field['g'] = psi_new_array

    dz = lambda A: d3.Differentiate(A, coords['z'])
    dx = lambda A: d3.Differentiate(A, coords['x'])
    u_x = dz(psi_field).evaluate(); u_x.change_scales(1)
    u_z = (-dx(psi_field)).evaluate(); u_z.change_scales(1)
    return u_x['g'], u_z['g']

def build_initial_condition(no_ice_checkpoint, no_ice_diags_dir,
                             Lx, Lz, nx, nz, Tm, eps, dealias=3/2,
                             T_top=0.0, h_min=0.02, smooth_frac=1/8,
                             n_last=3, out_file='initial_condition_mapped.h5'):

    Lz_old = Lz - Tm

    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=dealias)
    zbasis_old = d3.ChebyshevT(coords['z'], size=nz, bounds=(0, Lz_old), dealias=dealias)
    zbasis_new = d3.ChebyshevT(coords['z'], size=nz, bounds=(0, Lz), dealias=dealias)
    _, z_old = dist.local_grids(xbasis, zbasis_old)
    x_new, z_new = dist.local_grids(xbasis, zbasis_new)
    z_new_col = z_new.ravel()

    u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis_old))
    T = dist.Field(name='T', bases=(xbasis, zbasis_old))
    with h5py.File(no_ice_checkpoint, 'r') as h5f:
        u_data = h5f['tasks']['u'][-1]
        T_data = h5f['tasks']['T'][-1]
    u.change_scales(1); T.change_scales(1)
    u['g'][0] = u_data[0]; u['g'][1] = u_data[1]
    T['g'] = T_data

    x, z_full, psi_full, T_full = build_old_domain_fields(
        u, T, coords, xbasis, zbasis_old, dist, Lz_old, Lz, Tm, T_ice_top=T_top,
    )

    F_x = load_time_avg_heat_flux_top(no_ice_diags_dir, n_last=n_last)

    _, z0_x = ice_thickness_from_flux(
        F_x, Lz, Tm, T_top=T_top, h_min=h_min,
        smooth_modes=int(nx * smooth_frac), dist=dist, xbasis=xbasis,
    )
    h_x = z0_x - Lz_old

    # mh = np.mean(h_x)
    # h_xp = h_x - mh 
    # h_xp[abs(h_x)>(1 - Lz_old)] *= 0.5
    # print(h_xp)
    # # h_xp[:] *= 0.1
    # h_x = mh + h_xp
    
    psi_new, T_new = remap_piecewise_interface(
        z_full, psi_full, T_full, z_new_col, h_x, Lz_old, Lz,
    )
    ux_new, uz_new = differentiate_psi(psi_new, coords, xbasis, zbasis_new, dist)

    # mask velocity to zero inside ice, consistent with the model's f-damping
    mask = lambda zz: 0.5 * (1 + np.tanh(zz / (2 * eps)))
    z0_col = np.asarray(z0_x).reshape(-1, 1)
    f_new = mask(z_new - z0_col)
    ux_new *= (1 - f_new)
    uz_new *= (1 - f_new)

    with h5py.File(out_file, 'w') as h5f:
        h5f.create_dataset('psi', data=psi_new)        # (nx_ref, nz_ref) — NOT differentiated
        h5f.create_dataset('T', data=T_new)
        h5f.create_dataset('z0_x', data=z0_x.squeeze())
        h5f.create_dataset('x_ref', data=x_new.ravel())
        h5f.create_dataset('z_ref', data=z_new_col)
        h5f.attrs['Lx'], h5f.attrs['Lz'] = Lx, Lz
        h5f.attrs['Tm'], h5f.attrs['eps'] = Tm, eps

    print(f"Wrote {out_file}  (Lz_old={Lz_old:.4f})")


    fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(8,15))
    # c1 = ax1.contourf(np.squeeze(x), np.squeeze(z_full), psi_full.T)
    c1 = ax1.contourf(np.squeeze(x), np.squeeze(z_old), T['g'].T, levels=np.linspace(0, np.amax(T['g']), 11))
    ax1.contour(np.squeeze(x), np.squeeze(z_old), T['g'].T, levels=[0.2], colors='w')

    print(np.amin(T['g']))

    print(np.shape(ux_new))

    plt.colorbar(c1)
    # c2 = ax2.contourf(np.squeeze(x), np.squeeze(z_new), psi_new.T)
    c2 = ax2.contourf(np.squeeze(x), np.squeeze(z_new), T_new.T, levels=np.linspace(0, np.amax(T['g']), 11))
    ax2.axhline(0.8, color="r", linewidth=0.5)
    ax2.contour(np.squeeze(x), np.squeeze(z_new), T_new.T, levels=[0.2], colors='w', linewidths=0.4)
    
    plt.colorbar(c2)
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    ax3.plot(x, h_x)
    
    fig.savefig("remapped_conditions.png", dpi=400, bbox_inches="tight")

    return out_file


if __name__ == '__main__':
    # build_initial_condition(
    #     no_ice_checkpoint='no_ice_run/chkp/chkp_s1.h5',
    #     no_ice_diags_dir='no_ice_run/diags',
    #     Lx=4.0, Lz=1.0, nx=256, nz=128,
    #     Tm=0.0, eps=0.02)

    build_initial_condition("/home/hamish/Research/HorizontalConvection/lcl/remap_test/data/Ra8/000/chkp/chkp_s86.h5", "/home/hamish/Research/HorizontalConvection/lcl/remap_test/data/Ra8/000/diags/", 
                            5, 1, 512, 1024, 0.2, 2e-3, smooth_frac=20/512, n_last=40)