import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data range
x = np.linspace(-10, 10, 400)

# Linear equation: y = mx + b
y_linear = 2 * x + 3

# Quadratic equation: y = x^2 + 2x + 1
y_quadratic = x**2 + 2*x + 1

# Cubic equation: y = x^3 + x^2 + 2x + 1
y_cubic = x**3 + x**2 + 2*x + 1

# Quartic equation: y = x^4 + x^3 + 2x + 1
y_quartic = x**4 + x**3 + 2*x + 1

# Plotting
plt.figure(figsize=(12, 10))

# Linear
plt.subplot(2, 2, 1)
sns.lineplot(x=x, y=y_linear)
plt.title('Linear Equation')

# Quadratic
plt.subplot(2, 2, 2)
sns.lineplot(x=x, y=y_quadratic)
plt.title('Quadratic Equation')

# Cubic
plt.subplot(2, 2, 3)
sns.lineplot(x=x, y=y_cubic)
plt.title('Cubic Equation')

# Quartic
plt.subplot(2, 2, 4)
sns.lineplot(x=x, y=y_quartic)
plt.title('Quartic Equation')
plt.show()