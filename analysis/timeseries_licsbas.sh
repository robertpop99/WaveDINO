#!/bin/bash
# Supplementary S7: LiCSBAS time series for Laguna del Maule 083D from one correction variant (unw | gacos | wavedino).
# Inputs are built by analysis/timeseries_prep.py in $TS_ROOT (default ./timeseries). Needs LiCSBAS
# (https://github.com/comet-licsar/LiCSBAS) on the PATH.
# Usage: analysis/timeseries_licsbas.sh wavedino [start_step]   (start_step 11-16, default 11; resumes an interrupted run)

set -e
cd "${TS_ROOT:-timeseries}"

VARIANT=$1
START=${2:-11}
N_PARA=${N_PARA:-16}

CLIP=LdM_083D_${VARIANT}_CLIP
TS=LdM_083D_${VARIANT}_TS

[ 11 -ge $START ] && LiCSBAS11_check_unw.py -d $CLIP -t $TS

# pin the reference to pixel (x=460, y=213), the reference pixel of the GNSS evaluation (util.datasets.get_reference),
# so all variants share it
[ 11 -ge $START ] && echo "460:461/213:214" > $TS/info/120ref.txt

[ 12 -ge $START ] && LiCSBAS12_loop_closure.py -d $CLIP -t $TS --multi_prime --n_para $N_PARA --nopngs

[ 13 -ge $START ] && LiCSBAS13_sb_inv.py -d $CLIP -t $TS --mem_size 8000 --n_para $N_PARA --inv_alg WLS --nopngs

[ 14 -ge $START ] && LiCSBAS14_vel_std.py -t $TS --mem_size 16000

[ 15 -ge $START ] && LiCSBAS15_mask_ts.py -t $TS -i 300 -s 20 -c 0.04

[ 16 -ge $START ] && LiCSBAS16_filt_ts.py -t $TS --n_para $N_PARA --nopngs
