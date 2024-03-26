import rpy2.robjects as robjects
r = robjects.r
r['source']("moveHMM_model_fitting_and_statistics.R")
function_1 = robjects.globalenv["fit_model_Freda"]
result_1 = function_1()
print(result_1[1], result_1[2], result_1[3])

function_2 = robjects.globalenv["fit_model_Data"]
result_2 = function_2()
print(result_2)