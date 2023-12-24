import pandas as pd
from prf import load_all_prf_data

roi = load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "inferred_varea")
# size = load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "full-sigma")
ecc = load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "full-eccen")
betas = load_all_prf_data("F:\\ds003812-download\\derivatives\\processed\\betas","betas")

roi_result = []
# size_result = []
ecc_result = []
side_result = []
subj_result = []
betas_result = []

for freq in range(10):
    print(f"freq {freq}")
    for sub in range(12):    
        for side in (0,1):           
            if side < 1:
                betas_result.extend(betas[sub][freq,:len(ecc[sub][0])])
            else:
                betas_result.extend(betas[sub][freq,len(ecc[sub][0]):])

            roi_result.extend(roi[sub][side])
            #size_result.extend(size[sub][side])
            ecc_result.extend(ecc[sub][side])
            side_result.extend([side]*len(ecc[sub][side]))
            subj_result.extend([sub +1]*len(ecc[sub][side]))

    data = {'betas': betas_result,'roi':roi_result,'ecc':ecc_result,'side':side_result,'subj':subj_result}
    tab = pd.DataFrame(data)
    tab.to_csv(f"table_beta_freq_{freq}_data_full_inferred_varea.csv",index=False)


