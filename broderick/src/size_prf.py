import pandas as pd
from prf import load_all_prf_data

roi = load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "inferred_varea")
size = load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "full-sigma")
ecc = load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "full-eccen")

roi_result = []
size_result = []
ecc_result = []
side_result = []
subj_result = []

for sub in range(12):
    for side in (0,1):
        roi_result.extend(roi[sub][side])
        size_result.extend(size[sub][side])
        ecc_result.extend(ecc[sub][side])
        print(len(ecc[sub][side]))
        side_result.extend([side]*len(ecc[sub][side]))
        subj_result.extend([sub +1]*len(ecc[sub][side]))

data = {'roi':roi_result,'size':size_result,'ecc':ecc_result,'side':side_result,'subj':subj_result}

tab = pd.DataFrame(data)
tab.to_csv("table_size_prf_data_full_inferred_varea.csv",index=False)


