import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
# Create an array with 1000 values between -1.5 and 1.5
x = np.linspace(-1.5, 1.5, 1000)

# Define the periodic squared exponential function
def periodic_squared_exponential(x, l=1.0, p=1.0):
    return np.exp(-(np.sin(np.pi * x) ** 2))

# Compute the function values
y = periodic_squared_exponential(x)

# Plot the function with a specified figure size and increased resolution
plt.figure(figsize=(10, 2.5), dpi=100)
plt.plot(x, y)
#plt.title('Periodic Squared Exponential Function')
plt.xlabel('Normalized distance to position of the car')
plt.ylabel('Basis function value')
plt.legend(['Periodic Squared Exponential'], loc='upper right')
#plt.grid(True)
plt.tight_layout()
plt.show()
