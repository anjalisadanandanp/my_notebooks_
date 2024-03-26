#Fitting a moveHMM model
#---------------------------------------
#Packages required
require(moveHMM)
require(tseries)

fit_model_Freda <- function()   
{
    #Loading the data
    Freda <- read.csv("moveHMM_Elephant_data_minimum_24hrs.csv")

    #Pre-processing the data
    Freda_processed <- prepData(Freda, type="LL")

    #MODEL
    #step length --> "gamma"
    stepDist = c("gamma") 
    mu0 <- c(0.05,0.03)
    sigma0 <- c(0.1,0.03) 
    stepPar0 <- c(mu0,sigma0) 

    #turning angle --> "von Mises"
    angleDist = c("vm")
    angleMean0 <- c(pi,0)
    kappa0 <- c(1,1)
    anglePar0 <- c(angleMean0,kappa0)

    #fitting the model
    model <- fitHMM(data=Freda_processed, nbStates=2 , stepPar0=stepPar0 ,anglePar0=anglePar0 , stepDist=stepDist, angleDist=angleDist, formula=~1, stationary = TRUE)

    return (model)
}
#---------------------------------------

fit_model_Data <- function()
{

    Data <- read.csv("Freda_ll_utm.csv")

    #Pre-processing the data
    Data_processed <- prepData(Data)

    #MODEL
    #step length --> "gamma"
    stepDist = c("gamma") 
    mu0 <- c(0.05,0.03)
    sigma0 <- c(0.1,0.03) 
    stepPar0 <- c(mu0,sigma0) 

    #turning angle --> "von Mises"
    angleDist = c("vm")
    angleMean0 <- c(pi,0)
    kappa0 <- c(1,1)
    anglePar0 <- c(angleMean0,kappa0)

    #fitting the model
    model <- fitHMM(data=Data_processed, nbStates=2 , stepPar0=stepPar0 ,anglePar0=anglePar0 , stepDist=stepDist, angleDist=angleDist, formula=~1, stationary = TRUE)

    return (model)
}