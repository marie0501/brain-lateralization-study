from src.betas import get_all_betas
from src.prf import get_all_prf_data


def betas():

    directory = 'F:\\ds003812-download\\derivatives\\processed'
    filename = "sub-wlsub"
    
    get_all_betas(directory=directory,filename_prefix=filename, smooth=True)

def prf():

    area_directory = "F:\\ds003812-download\\derivatives\\prf_solutions\\sub-wlsubj001\\bayesian_posterior"
    ecc_directory = "F:\\ds003812-download\\derivatives\\prf_solutions\\sub-wlsubj001\\bayesian_posterior"

#get_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions","bayesian_posterior","inferred_sigma.mgz")
#get_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions","atlas","benson14_eccen.mgz")
get_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions","data","full-sigma.mgz")

