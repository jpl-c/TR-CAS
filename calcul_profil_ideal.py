from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


alpha = -np.pi/6
# tan_alpha = np.tan(alpha)
tan_alpha = 1.2
# print(tan_alpha)
H = 10

def df_pos (f, t):
    x = -t 
    dfdx = -(tan_alpha * x + H - f)/((H - f)*tan_alpha - x)
    return dfdx

def df_neg (f, t):
    x = -t 
    dfdx = -(-tan_alpha * x + H - f)/((H - f)*(-tan_alpha) - x)
    return dfdx

def var_profile_plot(H, top_vals):
    for top_val in top_vals:
        t_pos = np.linspace(0, H/tan_alpha, 101)
        t_neg = np.linspace(0, -H/tan_alpha, 101)
        t = np.concatenate((t_neg[::-1], t_pos))
        x0 = top_val
        sol_pos = odeint(df_pos, x0, t_pos)   
        sol_neg = sol_pos[::-1]
        sol = np.concatenate((sol_neg, sol_pos))
        plt.plot(t, sol, label = f"inital top value of {np.floor(top_val*100)/100}")

    i_max = 10
    theta_max = np.pi/6
    for i in range(i_max) :
        theta = i/i_max * theta_max
        ray_pos = [[0, 2*H*np.tan(theta)], [H, 0]]
        ray_neg = [[0, -2*H*np.tan(theta)], [H, 0]]
        plt.plot(ray_pos[0], ray_pos[1], ":k")
        plt.plot(ray_neg[0], ray_neg[1], ":k")

    Pos_buse = [0, H]
    plt.scatter(Pos_buse[0], Pos_buse[1])
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylim(bottom = 0)
    plt.axis("equal")
    plt.grid()
    plt.show()

top_vals = np.linspace(H/50, H/5, 5)
var_profile_plot(H, top_vals)







# x0 = 3
# t_pos = np.linspace(0, 2 * H/tan_alpha, 101)
# t_neg = np.linspace(0, -2 * H/tan_alpha, 101)

# sol_pos = odeint(df_pos, x0, t_pos)   
# sol_neg = odeint(df_neg, x0, t_neg)    
# plt.scatter(Pos_buse[0], Pos_buse[1])




# t = np.concatenate((t_neg[::-1], t_pos))
# sol = np.concatenate((sol_neg[::-1], sol_pos))

# plt.plot(t_pos, sol_pos[:], 'r', label = "f(x)")
# plt.plot(t_neg, sol_neg, 'r', label = "f(x)")
# plt.plot(t, sol, ":b",label = "f(x)")



