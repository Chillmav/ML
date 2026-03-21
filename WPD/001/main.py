from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


plots = 3
def f1(t, x):
    return x - 100

def f2(t, x):
    x1, x2 = x
    dx1 = x1 - 100
    dx2 = x2 - 200
    return [dx1, dx2]

def f3(t, x):

    x1, x2 = x
    dx1 = x1 - 100 + 50
    dx2 = x2 - 200 + 50
    return [dx1, dx2]

sol1 = solve_ivp(f1, [0, 1], [0], method='RK45')
sol2 = solve_ivp(f2, [0, 1], [0, 1], method="RK45")
sol3 = solve_ivp(f3, [0, 1], [0, 1], method="RK45")

solutions = [sol1, sol2, sol3]

plt.plot(solutions[0].t, solutions[0].y.T)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()

plt.show()

plt.plot(solutions[1].t, solutions[1].y.T)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()

plt.show()

plt.plot(solutions[2].t, solutions[2].y.T)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.show()



