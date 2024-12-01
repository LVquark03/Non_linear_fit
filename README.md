### Non_linear_fit

The Fit.py script provides a collection of functions for performing linear and non-linear curve fitting using Python libraries like matplotlib and scipy. It includes functions to fit data points to a given function, calculate and print fitting parameters with uncertainties, perform chi-squared tests, and visualize the fit with plots. Key functions include fit, lin_fit, lin_fit_err, and fit_plot.

###How to use
To perform a non-linear fit, you can use the fit function with a custom function:

from Fit import fit

# Define a custom function
func = lambda x, a, b: a * x + b

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
sy = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
guess = [1, 1]

# Perform non-linear fit
popt, cov = fit(func, x, y, sy, guess)
Plotting the Fit

To visualize the fit, use the fit_plot function:

from Fit import fit_plot

# Perform the fit first to obtain popt
popt, _ = fit(func, x, y, sy, guess)

# Plot the fit
fit_plot(func, x, y, sy, popt, title='Fit Plot', ylabel='y', xlabel='x')



