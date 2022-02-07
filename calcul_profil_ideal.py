from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


tan_alpha = 1.2
H = 5

def df_pos_dec (f, t):
    x = -(t +H/tan_alpha) 
    dfdx = -(tan_alpha * x + H - f)/((H - f)*tan_alpha - x)
    return dfdx

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
        ray_pos = [[0, H*np.tan(theta)], [H, 0]]
        ray_neg = [[0, -H*np.tan(theta)], [H, 0]]
        plt.plot(ray_pos[0], ray_pos[1], ":k")
        plt.plot(ray_neg[0], ray_neg[1], ":k")

    Pos_buse = [0, H]
    plt.scatter(Pos_buse[0], Pos_buse[1])
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylim(bottom = 0)
    plt.axis("equal")
    plt.grid()

# top_vals = np.linspace(1.4, 1.8, 1)
# var_profile_plot(H, top_vals)

def plot_profil_ideal(H, tan_alpha):
    x0 = 0
    max_x = H/tan_alpha
    t_pos = np.linspace(-max_x, 0, 100)
    t_neg = np.linspace(0, max_x, 100)
    sol = odeint(df_pos_dec, x0, t_pos)
    sol= sol -sol[-1]
    t = np.concatenate((t_neg-max_x, t_pos+max_x))
    sol = np.concatenate((sol[::-1], sol))
    plt.plot(t, sol)

    # i_max = 10
    # theta_max = np.arctan(tan_alpha)
    # for i in range(i_max+1) :
    #     theta = i/i_max * theta_max
    #     ray_pos = [[0, H*np.tan(theta)], [H, 0]]
    #     ray_neg = [[0, -H*np.tan(theta)], [H, 0]]
    #     plt.plot(ray_pos[0], ray_pos[1], ":k")
    #     plt.plot(ray_neg[0], ray_neg[1], ":k")

    Pos_buse = [0, H]
    plt.scatter(Pos_buse[0], Pos_buse[1])
    plt.axis("equal")
    plt.grid()
    plt.show()


plot_profil_ideal(H, tan_alpha)






