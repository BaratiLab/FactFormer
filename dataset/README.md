### 2D Kolmogorov flow

The script for generating 2D Kolmogorov flow:```kolmogorov.py``` is modified from the original script from: https://github.com/neuraloperator/physics_informed/tree/master/solver.

The modification is that we added a drag force term following [Kochkov et al.](https://www.pnas.org/doi/10.1073/pnas.2101784118) and a slightly more complex forcing term: $8\cos(8y)$ as opposed 
to $4\cos(4y)$ to generate more complex pattern (i.e. more vorticies).

### 3D Isotropic turbulence

Installation of [SpectralDNS](https://github.com/spectralDNS/spectralDNS) is required to run the script. For the example of bash command, see: ```run_iso.sh```.

### 3D Smoke

The script for generating 3d buoyancy-driven flow is modified from the 2D script from [PDEArena](https://github.com/microsoft/pdearena),
installation of [PhiFlow](https://github.com/tum-pbs/PhiFlow) is required. 
