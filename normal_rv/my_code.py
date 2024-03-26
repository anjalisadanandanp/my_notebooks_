#create a plot of a normal distribution

#import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#function to plot a normal distribution
#parameters: mean, standard deviation
def plot_normal(mean, std):

    #create a line plot of the normal distribution
    x = np.linspace(mean - 4*std, mean + 4*std, 100)
    plt.plot(x, stats.norm.pdf(x, mean, std), color='r')

    #set the title
    plt.title('Normal Distribution')

    #set the x-axis label
    plt.xlabel('X')

    #set the y-axis label
    plt.ylabel('Probability Density')

    #gridlines
    plt.grid(True)

    #show the plot
    plt.savefig('normal_rv/normal_'+ str(mean) + "_" + str(std) +'_.png')

#call the function
plot_normal(60, 20)


