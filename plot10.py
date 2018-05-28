import matplotlib.pyplot as plt
import numpy as np

y1 = [98.15, 93.26666666666667, 87.0, 84.72, 81.55, 82.74285714285715, 83.125, 84.24444444444444, 85.18]
y01 = [ 98.5, 91.0, 82.05, 80.24, 75.6, 77.2, 77.15, 79.05555555555556, 79.73]


x1 = np.linspace(2, 10, 9)
y_bound = [94.1 for i in range(10)]
x_icarl = np.linspace(2, 10, 10)

print(type(y1))
plt.plot(x1,y1,'-', linewidth=2.0, label=r'$\lambda=1$')
plt.plot(x1,y01,'-', linewidth=2.0, label=r'$\lambda=0.01$')
plt.plot(x_icarl,y_bound,'r--', linewidth=2.0, label='Accuracy upper bound for 100 class')
plt.ylabel('Accuracy (%)')
plt.xlabel('Number of classes')
plt.grid(True)
plt.legend()
plt.show()
#plot(x01, y, 'bo')

