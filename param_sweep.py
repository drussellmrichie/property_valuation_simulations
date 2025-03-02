import core_simulation as cs
import random
import numpy as np
from itertools import product
import os

grid_size          = 100              # number of properties in the x and y dimension, grid_size = 100 makes 100 by 100 grid.
true_price_per_isq = 4                # "dollars" per improved square foot
lv_center          = 100000
lv_decay           = .03

numb_sims           = 50

lv_noise_sds      = [1, 10]           # should really be called lv_noise_factor but keeping name for legacy reasons
isq_from_lvs      = [True]            # whether improved square footage should be generated from land values, or generated independently (with same process as land values)
isq_noise_sds     = [.25, .01]        # should really be called isq_noise_factor but keeping name for legacy reasons

iv_noise_factors  = [.3, .15]
iv_noise_sds      = [None]            # just a placeholder

tv_obs_noise_sds  = [2.5e-2, 1]
train_n           = 3

max_tries  = 10

prod_gen = product(lv_noise_sds, isq_from_lvs, isq_noise_sds, iv_noise_factors, iv_noise_sds, tv_obs_noise_sds)

folder = '../results/key_conditions_tv_noise_iv_noise_fixed/extreme_values'

existing_files = os.listdir(folder)

for lv_noise_sd, isq_from_lv, isq_noise_sd, iv_noise_factor, iv_noise_sd, tv_obs_noise_sd in prod_gen:
    for sim_numb in range(numb_sims):
        fname = f"{grid_size}_{true_price_per_isq}_{tv_obs_noise_sd}_{lv_noise_sd}_{isq_from_lv}_{isq_noise_sd}_{iv_noise_sd}_{iv_noise_factor}_{train_n}_{sim_numb}"

        if f'{fname}_eval_metrics.csv' in existing_files:
            print(f'{fname} already exists. continuing to next simulation.')
            continue

        finished   = False
        numb_tries = 0
        while not finished and numb_tries < max_tries:
            numb_tries += 1
            print(numb_tries)
            try:
                df_true = cs.make_data(grid_size, true_price_per_isq, lv_center, lv_decay, lv_noise_sd,
                                    isq_noise_sd, iv_noise_factor, iv_noise_sd, tv_obs_noise_sd, isq_from_lv)

                morans     = cs.calculate_spatial_autocorr(df_true)
                df_test    = cs.fit_data(df_true, train_n)
                results_df = cs.evaluate_model(df_test)
                finished = True
            except np.linalg.LinAlgError:
                pass
        
        with open(f'{folder}/{fname}_numb_tries={numb_tries}.txt', 'w') as f:
            f.write(str(numb_tries))

        morans.to_csv(    f'{folder}/{fname}_morans.csv')
        df_test.to_csv(   f'{folder}/{fname}_df_test.csv')        
        results_df.to_csv(f'{folder}/{fname}_eval_metrics.csv')
