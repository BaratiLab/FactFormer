nn=0

while [ $nn -le 1100 ]; do
    echo "$nn"

    mpirun -np 10 \
    python Isotropic.py \
    --dt 0.002 \
    --T 1 \
    --write_result 25 \
    --compute_spectrum 200 \
    --compute_energy 100 \
    --N 60 60 60 \
    --kd 50 \
    --Re_lam 84 \
    --integrator RK4 \
    --optimization numba \
    --out_folder iso_turb_results \
    --postfix "$nn" \
    --random_seed $nn \
    NS

    nn=$(expr $nn + 1)
done
