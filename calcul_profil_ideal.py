from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


alpha = np.pi/6
tan_alpha = np.tan(alpha)
# print(tan_alpha)
H = 3

def df (f, t):
    x = t - H*tan_alpha
    dfdx = (tan_alpha * x + H - f)/(H - f*tan_alpha - x)
    return dfdx

x0 = 0

t = np.linspace(0, 5, 101)

sol = odeint(df, x0, t)    

plt.plot(t, sol[:], 'g', label = "f(x)")
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()