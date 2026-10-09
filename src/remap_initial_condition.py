#!/usr/bin/env python
"""
Command-line wrapper around build_initial_condition.

Assumes the standard run layout, where chkp/ and diags/ are both
subdirectories of one run directory:

    <run_dir>/chkp/chkp_s*.h5
    <run_dir>/diags/diags_s*.h5

The last checkpoint (by simulation-write number) is used automatically.

Example (equivalent to the previous hardcoded call):

    python run_build_initial_condition.py \
        /home/hcfch1/scratch/HorizontalConvection/lcl/A5_Ra8_no_phase/data/Ra8/000 \
        --Lx 5 --Lz 1 --nx 512 --nz 1024 --Tm 0.2 --eps 2e-3 \
        --smooth_frac 0.0390625 \
        --n_last 40

Only the run directory is required; every physical/numerical parameter
below has a default and can be overridden individually.
"""
import argparse
import glob
import os
import re
from build_remapped_initial_condition import build_initial_condition


def find_last_checkpoint(run_dir):
    """
    Return the path to the highest-numbered chkp_s*.h5 file under
    <run_dir>/chkp/. Sorts numerically on the trailing write number
    rather than lexicographically, so e.g. chkp_s9.h5 vs chkp_s10.h5
    is handled correctly.
    """
    chkp_dir = os.path.join(run_dir, 'chkp')
    candidates = glob.glob(os.path.join(chkp_dir, 'chkp_s*.h5'))
    if not candidates:
        raise FileNotFoundError(f"No chkp_s*.h5 files found in {chkp_dir}")

    def write_number(path):
        match = re.search(r'chkp_s(\d+)\.h5$', os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(candidates, key=write_number)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a remapped (psi, T) initial condition for a "
                    "phase-field ice run from an equilibrated no-ice checkpoint."
    )

    # Required positional arg: the run directory (contains chkp/ and diags/)
    parser.add_argument('run_dir', type=str,
                        help="Path to the no-ice run directory, containing "
                             "chkp/ and diags/ subdirectories")

    # Everything else is optional, with sensible defaults
    parser.add_argument('--Lx', type=float, default=5.0,
                        help="Domain width (default: 5.0)")
    parser.add_argument('--Lz', type=float, default=1.0,
                        help="Domain height, phase-field run (default: 1.0)")
    parser.add_argument('--nx', type=int, default=512,
                        help="Number of x grid points (default: 512)")
    parser.add_argument('--nz', type=int, default=1024,
                        help="Number of z grid points (default: 1024)")
    parser.add_argument('--Tm', type=float, default=0.2,
                        help="Melt temperature (default: 0.2)")
    parser.add_argument('--eps', type=float, default=2e-3,
                        help="Phase-field interface thickness (default: 2e-3)")
    parser.add_argument('--dealias', type=float, default=3/2,
                        help="Dealiasing factor (default: 1.5)")
    parser.add_argument('--T_top', type=float, default=0.0,
                        help="Temperature at the domain top (default: 0.0)")
    parser.add_argument('--h_min', type=float, default=0.02,
                        help="Minimum ice thickness floor (default: 0.02)")
    parser.add_argument('--smooth_frac', type=float, default=1/8,
                        help="Fraction of x-modes retained when smoothing "
                             "the flux-derived thickness profile "
                             "(default: 0.125)")
    parser.add_argument('--n_last', type=int, default=3,
                        help="Number of trailing time-averaged flux writes "
                             "to average together (default: 3)")
    parser.add_argument('--n_end', type=int, default=3,
                        help="Number of trailing time-averaged flux writes "
                             "to average together (default: 3)")
    parser.add_argument('--center_flux_max', type=int, default=0,
                        help="Center the heat flux at domain middle")
    parser.add_argument('--out_file', type=str,
                        default='initial_condition_mapped.h5',
                        help="Output .h5 filename "
                             "(default: initial_condition_mapped.h5)")

    return parser.parse_args()


def main():
    args = parse_args()

    no_ice_checkpoint = find_last_checkpoint(args.run_dir)
    no_ice_diags_dir = os.path.join(args.run_dir, 'diags')
    print(f"Using checkpoint: {no_ice_checkpoint}")
    print(f"Using diags dir:  {no_ice_diags_dir}")

    build_initial_condition(
        no_ice_checkpoint,
        no_ice_diags_dir,
        Lx=args.Lx,
        Lz=args.Lz,
        nx=args.nx,
        nz=args.nz,
        Tm=args.Tm,
        eps=args.eps,
        dealias=args.dealias,
        T_top=args.T_top,
        h_min=args.h_min,
        smooth_frac=args.smooth_frac,
        n_last=args.n_last,
        n_end=args.n_end,
        out_file=args.out_file,
        center_flux_max=args.center_flux_max
    )


if __name__ == '__main__':
    main()