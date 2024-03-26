#Choosing the best fitted model based on AIC scores
#models are initialized with different starting points

require(moveHMM)
require(tseries)

#Loading the data
Freda <- read.csv("/home/anjalip/Documents/My_codes/rpy2_tutorials/moveHMM_Elephant_data_minimum_24hrs.csv")

#Pre-processing the data
Freda_processed <- prepData(Freda, type="LL") 


######################################################################################################
#function to fit the model
fit_movement_model <- function(stepDist, mu0, sigma0, angleDist, angleMean0, kappa0) {
  stepPar0 <- c(mu0,sigma0)
  anglePar0 <- c(angleMean0,kappa0)
  movement_model <- fitHMM(data=Freda_processed, nbStates=2 , stepPar0=stepPar0 ,anglePar0=anglePar0 , stepDist=stepDist, angleDist=angleDist, formula=~1, stationary = TRUE)
  return (movement_model)
}
######################################################################################################


######################################################################################################
#model with best fit starting condition
stepDist = c("gamma")
mu0 <- c(0.01,0.003)
sigma0 <- c(0.0003,0.03)
angleDist = c("vm")
angleMean0 <- c(3.14,0)
kappa0 <- c(0.3,1.5)

#fit the model
model_best_fit <- fit_movement_model(stepDist, mu0, sigma0, angleDist, angleMean0, kappa0)



getPalette_modified <- function(nbStates) {
  if(nbStates < 8) {
    # color-blind friendly palette
    #pal <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    pal <- c("red", "blue")
    col <- pal[1:nbStates]
  } else {
    # to make sure that all colours are distinct (emulate ggplot default palette)
    hues <- seq(15, 375, length = nbStates + 1)
    col <- hcl(h = hues, l = 65, c = 100)[1:nbStates]
  }
  return(col)
}

plot_distributions <- function(x, animals = NULL, ask = TRUE, breaks = "Sturges", col = NULL,
                         plotTracks = TRUE, plotCI = FALSE, alpha = 0.95, ...) {
  m <- x # the name "x" is for compatibility with the generic method
  nbStates <- ncol(m$mle$stepPar)
  
  # prepare colours for the states (used in the maps and for the densities)
  if(is.null(col) | (!is.null(col) & length(col) != nbStates)) {
    col <- getPalette_modified(nbStates = nbStates)
  }
  
  #################################
  ## State decoding with Viterbi ##
  #################################
  if(nbStates > 1) {
    cat("Decoding states sequence... ")
    states <- viterbi(m)
    cat("DONE\n")
  } else {
    states <- rep(1,nrow(m$data))
  }
  
  ########################################
  ## Plot state-dependent distributions ##
  ########################################
  par(mar = c(4, 4, 0, 0)) # bottom, left, top, right
  par(mgp=c(1, 1, 1))
  par(ask = FALSE)
  
  distData <- getPlotData(m = m, type = "dist")
  
  # setup line options
  legText <- c("Encamped", "Exploratory", "Total")
  lty <- c(rep(1, nbStates), 2)
  lwd <- c(rep(2, nbStates), 2)
  lineCol <- c(col, "black")
  
  print("LINEWEIGHTS:")
  print(lwd)
  
  cex_value = 1
  
  png("/home/anjalip/Documents/My_codes/rpy2_tutorials/two_states/step_length_distributions.png", 
      units="in", width=5, height=5, res=300, pointsize=12)
  
  par(cex = cex_value)
  par(cex.axis = cex_value)
  par(cex.lab=cex_value)
  par(cex.main = cex_value)
  
  # define ymax for step histogram
  h <- hist(m$data$step, plot = FALSE, breaks = breaks)
  ymax <- 1.3 * max(h$density)
  maxdens <- max(distData$step$total)
  if(maxdens > ymax & maxdens < 1.5 * ymax) {
    ymax <- maxdens
  }
  
  # step length histogram
  hist(m$data$step, prob = TRUE, main = "",
       xlab = "Step length (km)", col = "lightgrey", border = "white",
       breaks = breaks, cex.axis = cex_value, cex.lab=cex_value, cex.main=cex_value, xlim=c(0,0.1), ylim=c(0, 100))
  for(i in 1:(nbStates + 1)) {
    lines(distData$step$step, distData$step[,i+1], col = lineCol[i],
          lty = lty[i], lwd = lwd[i])
  }
  
  legend("top", legText, lwd = lwd, col = lineCol, lty = lty, bty = "y")
  
  # define ymax and breaks for angle histogram
  h1 <- hist(m$data$angle, plot = FALSE, breaks = breaks)
  breaks <- seq(-pi, pi, length = length(h1$breaks))
  h2 <- hist(m$data$angle, plot = FALSE, breaks = breaks)
  ymax <- 1.3 * max(h2$density)

  dev.off()
  

  png("/home/anjalip/Documents/My_codes/rpy2_tutorials/two_states/turning_angle_distributions.png", 
      units="in", width=5, height=5, res=300, pointsize=12)
  par(cex = cex_value)
  par(cex.axis = cex_value)
  par(cex.lab=cex_value)
  par(cex.main = cex_value)
  
  # turning angle histogram
  hist(m$data$angle, ylim = c(0, ymax), prob = TRUE, main = "",
       xlab = "Turning angle (radians)", col = "lightgrey", border = "white",
       breaks = breaks, xaxt = "n", cex.axis = cex_value, cex.lab=cex_value, cex.main=cex_value)
  axis(1, at = c(-pi, -pi/2, 0, pi/2, pi),
       labels = expression(-pi, -pi/2, 0, pi/2, pi))
  for(i in 1:(nbStates + 1)) {
    lines(distData$angle$angle, distData$angle[,i+1], col = lineCol[i],
          lty = lty[i], lwd = lwd[i])
  }
  
  legend("top", legText, lwd = lwd, col = lineCol, lty = lty, bty = "y")
  dev.off()
  
  ##################################################
  ## Plot the t.p. as functions of the covariates ##
  ##################################################
  
  beta <- m$mle$beta
  if(nbStates > 1 & nrow(beta) > 1) {
    trProbs <- getPlotData(m, type = "tpm", format = "wide")
    
    # loop over covariates
    par(mfrow = c(nbStates, nbStates))
    par(mar = c(5, 4, 4, 2) - c(0, 0, 1.5, 1)) # bottom, left, top, right
    for(cov in 1:ncol(m$rawCovs)) {
      trProbsCov <- trProbs[[cov]]
      # loop over entries of the transition probability matrix
      for(i in 1:nbStates) {
        for(j in 1:nbStates) {
          trName <- paste0("S", i, "toS", j)
          plot(trProbsCov[,1], trProbsCov[,trName], type = "l",
               ylim = c(0, 1), xlab = names(trProbs)[cov],
               ylab = paste(i, "->", j))
          
          # derive confidence intervals using the delta method
          if(plotCI) {
            options(warn = -1) # to muffle "zero-length arrow..." warning
            # plot the confidence intervals
            arrows(trProbsCov[,1], trProbsCov[,paste0(trName, ".lci")],
                   trProbsCov[,1], trProbsCov[,paste0(trName, ".uci")],
                   length = 0.025, angle = 90, code = 3,
                   col = gray(0.5), lwd = 0.7)
            options(warn = 1)
          }
        }
      }
      
      mtext("Transition probabilities", side = 3, outer = TRUE, padj = 2)
    }
  }
  
  #################################
  ## Plot maps colored by states ##
  #################################
  # Prepare the data
  nbAnimals <- length(unique(m$data$ID))
  if(is.character(animals)) {
    if(any(!animals %in% unique(m$data$ID))) {
      stop("Check 'animals' argument, ID not found")
    }
    animalsInd <- which(unique(m$data$ID) %in% animals)
  } else if(is.numeric(animals)) {
    if(min(animals) < 1 | max(animals) > nbAnimals) {
      stop("Check 'animals' argument, index out of bounds")
    }
    animalsInd <- animals
  } else {
    animalsInd <- 1:nbAnimals
  }
  nbAnimals <- length(animalsInd)
  ID <- unique(m$data$ID)[animalsInd]
  
  if(nbStates>1 & plotTracks) { # no need to plot the map if only one state
    par(mfrow = c(1, 1))
    
    for(zoo in 1:1) {
      
      png(paste(c("/home/anjalip/Documents/My_codes/rpy2_tutorials/two_states/animal_ID", zoo, ".png"), collapse = "_"), 
          units="in", width=4.8, height=4.8, res=300, pointsize=12)
      
      par(cex = cex_value)
      par(cex.axis = cex_value)
      par(cex.lab=cex_value)
      par(cex.main = cex_value)
      
      # data for this animal
      ind <- which(m$data$ID == ID[zoo])
      s <- states[ind]
      x <- m$data$x[ind]
      y <- m$data$y[ind]
      
      #choose states with 1
      index_01 <- which(states == 1 & m$data$ID == ID[zoo])
      x_01 = x[index_01]
      y_01 = y[index_01]
      
      index_02 <- which(states == 2 & m$data$ID == ID[zoo])
      x_02 = x[index_02]
      y_02 = y[index_02]
      
      # slightly different for 2D and 1D data
      if(!all(y == 0)) {
        plot(x, y, pch = 16, col = col[s], cex = 0.75, asp = 1, xlab = "Longitude", ylab = "Latitude", cex.axis = cex_value, cex.lab=cex_value, cex.main=cex_value)
        #plot(x_02, y_02, pch = 16, col = "blue", cex = 1, asp = 1,
             #xlab = "Longitude", ylab = "Latitude", cex.axis = cex_value, cex.lab=cex_value, cex.main=cex_value)
        plot(x_01, y_01, pch = 16, col = "red", cex = 1, asp = 1,
             xlab = "Longitude", ylab = "Latitude", cex.axis = cex_value, cex.lab=cex_value, cex.main=cex_value)
        segments(x0 = x[-length(x)], y0 = y[-length(y)],
                 x1 = x[-1], y1 = y[-1],
                 col = col[s[-length(s)]], lwd = 2)

      } else { # if 1D data
        plot(x, xlab = "time", ylab = "x", pch = 16,
             cex = 0.5, col = col[s])
        segments(x0 = 1:(length(x) - 1), y0 = x[-length(x)],
                 x1 = 2:length(x), y1 = x[-1],
                 col = col[s[-length(x)]], lwd = 1.3)
      }
    
    # Add legend manually
    legend("topright", legend = c("Encamped", "Exploratory"), 
           pch = c(16, 16), col = c("red", "blue"))
      
    #save the plot
    dev.off()
    }
  }
  
  # set the graphical parameters back to default
  par(mfrow = c(1, 1))
  par(mar = c(5, 4, 4, 2) + 0.1) # bottom, left, top, right
  par(ask = FALSE)
  
}



plot_distributions(
  model_best_fit,
  animals = NULL,
  ask = TRUE,
  breaks = "Sturges",
  col = NULL,
  plotTracks = TRUE,
  plotCI = TRUE,
  alpha = 0.95
)












print(AIC(model_best_fit))
######################################################################################################

#1
######################################################################################################
print(model_best_fit)
out<-capture.output(model_best_fit)
cat(out,file="/home/anjalip/Documents/My_codes/rpy2_tutorials/two_states/model_params.txt",sep="\n",append=TRUE)
######################################################################################################

#2
######################################################################################################
#Plot the trajectories
plot(Freda_processed, compact = TRUE)
######################################################################################################

#3
######################################################################################################
#state probabilities (local decoding)
sp <- stateProbs(model_best_fit) 
write.csv(sp, "/home/anjalip/Documents/My_codes/rpy2_tutorials/two_states/two_state_HMM_stateProbs.csv")
######################################################################################################

#4
######################################################################################################
#state probabilities (global decoding)
states <- viterbi(model_best_fit) 
write.csv(states, "/home/anjalip/Documents/My_codes/rpy2_tutorials/two_states/two_state_HMM_viterbi.csv")
######################################################################################################

#5
######################################################################################################
#plot states and state probabilities in an observation (both viterbi and stateProbs)
plotStates(model_best_fit) 
######################################################################################################


#6
######################################################################################################
#Plotting the fitted model
plot(model_best_fit, CI=TRUE)
######################################################################################################



#7
######################################################################################################
#model checking
plotPR(model_best_fit)
######################################################################################################

