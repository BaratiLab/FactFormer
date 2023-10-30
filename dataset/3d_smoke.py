# modified from:
#: https://github.com/microsoft/pdearena

import logging
import os

import numpy as np
import torch 

from joblib import Parallel, delayed
from phi.flow import ( # SoftGeometryMask,; Sphere,; batch,; tensor, 
    Box, 
    CenteredGrid,
    Noise, 
    StaggeredGrid,
    advect,
    diffuse, 
    extrapolation,
    fluid, ) 
from phi.math import reshaped_native 
from phi.math import seed as phi_seed
from tqdm import tqdm
import logging
import sys
import timeit
from functools import partialmethod
from typing import Tuple

logger = logging.getLogger(__name__)


class Timer:
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


def generate_trajectories_smoke(
    num_samples: int,
    dirname: str = "data",
    n_parallel: int = 1,
    seed: int = 42,
) -> None:
    """
    Generate data trajectories for smoke inflow in bounded domain
    """
    # currently data is hard coded to be 64x64x64
    # in addition, the time interval is hard coded to be 1 (equals to physical dt=1 seconds)
    # each trajectory length is hardcoded to be 30

    logger.info(f"Number of samples: {num_samples}")


    def genfunc(idx, s):
        s = s + idx
        phi_seed(s) 

        if os.path.exists(os.path.join(dirname, f"smoke_{idx}.npz")):
            return

        res = 64

        smoke = abs(
            CenteredGrid( 
                Noise(scale=15.0, smoothness=4.0), 
                extrapolation.BOUNDARY, 
                x=res, 
                y=res, 
                z=res, 
                bounds=Box['x,y,z', 0:8.0, 0:8.0, 0:8.0], ) 
            ) # sampled at cell centers "
        velocity = StaggeredGrid(0, extrapolation.ZERO, x=res, y=res, z=res, bounds=Box['x,y,z', 0 : 8.0, 0 : 8.0, 0: 8.0])
        sub = res//64

        fluid_field_ = [] 
        velocity_ = []
        NT = 20
        skip_nt = 8
        dt = 0.75
        buoyancy_z = 0.5
        nu = 0.003

        for i in range(0, NT + skip_nt): 
            smoke = advect.semi_lagrangian(smoke, velocity, dt)
            buoyancy_force = (smoke * (0, 0, buoyancy_z)).at(velocity) # resamples smoke to velocity sample points 
            velocity = advect.semi_lagrangian(velocity, velocity, dt) + dt * buoyancy_force
            velocity = diffuse.explicit(velocity, nu, dt)
            velocity, _ = fluid.make_incompressible(velocity)
            
            if i >= skip_nt:
                fluid_field_.append(reshaped_native(smoke.values,
                                                    groups=("x", "y", "z", "vector"), 
                                                    to_numpy=True)[::sub, ::sub, ::sub])
                velocity_.append(
                        reshaped_native(
                        velocity.staggered_tensor(),
                        groups=("x", "y", "z", "vector"),
                        to_numpy=True, 
                        )[:-1:sub, :-1:sub, :-1:sub]
                        )
        
        fluid_field_ = np.stack(fluid_field_, axis=0)
        velocity_ = np.stack(velocity_, axis=0)
        assert fluid_field_.shape == (NT, 64, 64, 64, 1)
        assert velocity_.shape == (NT, 64, 64, 64, 3)

        np.savez(os.path.join(dirname, f"smoke_{idx}.npz"), fluid_field=fluid_field_, velocity=velocity_)

    os.makedirs(dirname, exist_ok=True)

    with Timer() as gentime:
        rngs = np.random.randint(np.iinfo(np.int32).max, size=num_samples)
        Parallel(n_jobs=n_parallel)(delayed(genfunc)(idx, rngs[idx]) for idx in tqdm(range(num_samples)))
        

    logger.info(f"Took {gentime.dt:.3f} seconds")
            

if __name__ == "__main__":
    generate_trajectories_smoke(8, "smoke_data_nu0.003_b0.5_lowres", n_parallel=8, seed=1234)