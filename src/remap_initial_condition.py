#!/usr/bin/env python
"""
Command-line wrapper around build_initial_condition.

Example (equivalent to the previous hardcoded call):

    python run_build_initial_condition.py \
        /home/hcfch1/scratch/HorizontalConvection/lcl/A5_Ra8_no_phase/data/Ra8/000/chkp/chkp_s210.h5 \
        /home/hcfch1/scratch/HorizontalConvection/lcl/A5_Ra8_no_phase/data/Ra8/000/diags/ \
        --Lx 5 --Lz 1 --nx 512 --nz 1024 --Tm 0.2 --eps 2e-3 \
        --smooth_frac 0.0390625 \
        --n_last 40

Only the two file paths are required; every physical/numerical parameter
below has a default and can be overridden individually.
"""
import sys 
sys.path.append("/home/hcfch1/scratch/HorizontalConvection/src/")
import argparse
from build_remapped_initial_condition import build_initial_condition


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a remapped (psi, T) initial condition for a "
                    "phase-field ice run from an equilibrated no-ice checkpoint."
    )

    # Required positional args: just the two file paths
    parser.add_argument('no_ice_checkpoint', type=str,
                        help="Path to the no-ice run's checkpoint .h5 file")
    parser.add_argument('no_ice_diags_dir', type=str,
                        help="Path to the no-ice run's diags directory "
                             "(for the time-averaged top heat flux)")

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
    parser.add_argument('--out_file', type=str,
                        default='initial_condition_mapped',
                        help="Output filename (do not append with .h5) "
                             "(default: initial_condition_mapped)")

    return parser.parse_args()


def main():
    args = parse_args()

    build_initial_condition(
        args.no_ice_checkpoint,
        args.no_ice_diags_dir,
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
        out_file=args.out_file,
    )


if __name__ == '__main__':
    main()