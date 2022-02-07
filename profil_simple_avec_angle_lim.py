from cProfile import Profile
from dis import dis
from hashlib import new
from math import dist, prod
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import intersect
import scipy.interpolate
import scipy.optimize
from scipy.signal import savgol_filter


def rotation_lim(theta, Pos_buse, profile = [], max_reflexion_angle = np.pi/6):
    """profile is a 2d list. profile[0] is the x values of each point of the profile. profile[1] is a list 
    containing the height values of the profile for each x"""

    L_x = np.linspace(profile[0][0], profile[0][-1], len(profile[0]))
    if len(profile) == 0 :
        # Setting up a basic profile if none was submitted
        psi_max = 0.7
        sigma_p = 0.5
        mu_p  = 0
        f = lambda t : 1/(sigma_p* np.sqrt(2*np.pi)) * np.exp(- (t-mu_p)**2/(2*sigma_p**2))
        L_x = np.linspace(-3, 3, 100)
        profile = np.array([L_x, f(L_x)])

    w_length = len(profile[0])//2
    if w_length % 2 == 0:
        w_length += 1


    ### Smoothing the given profile
    profile[0], profile[1] = savgol_filter((profile[0], profile[1]), window_length=w_length, polyorder=3)

    ### Creating interpolated function of the profile
    f_profile = scipy.interpolate.interp1d(profile[0], profile[1], kind = "cubic", fill_value='extrapolate')

    ### Determining the intersection point of the primary ray of the nozzle and the profile
    func_x0 = lambda x0 : x0-Pos_buse[0] + (f_profile(x0) - H)*np.tan(theta)
    x0 = scipy.optimize.fsolve(func_x0, (Pos_buse[0] + H*np.tan(theta))/2)[0] 
    y0 = f_profile(x0)

    ### Determining the sigma value for the virtual gaussian curve
    dist_head_curve = np.sqrt((Pos_buse[0] - x0)**2 + (Pos_buse[1] - y0)**2)
    h = dist_head_curve

    psi_max = np.pi/10 # Max spray angle of the nozzle

    sigma = h*np.tan(psi_max)/5
    mu = 0
    G = lambda t : 1/(sigma* np.sqrt(2*np.pi)) * np.exp(- (t-mu)**2/(2*sigma**2))

    ### Calculating the slope values of the profile
    slopes = []
    for i in range(len(profile[0])-1):
        xi = profile[0][i]
        xi1 = profile[0][i+1]
        yi = profile[1][i]
        yi1 = profile[1][i+1]
        if xi-xi1 != 0:
            slope =  (yi1 - yi)/(xi1 - xi)
            slopes.append(np.floor(slope*1000)/1000)
    slopes.append(slope)

    d_profile_hat = [[],[]]
    d_profile_hat[0], d_profile_hat[1] = savgol_filter((profile[0], slopes), window_length=w_length, polyorder=5)
    d_profile = scipy.interpolate.interp1d(d_profile_hat[0], d_profile_hat[1], kind = "cubic", fill_value='extrapolate')


    def effeciency(tan_phi):
        k = 4.5
        tan_phi_max = 1.2
        if tan_phi**2 < tan_phi_max**2:
            return (np.exp(-k*(tan_phi/tan_phi_max)**2) - np.exp(-k))/(1-np.exp(-k))
        else :
            # print("ricochet")
            return 0

    n_ray = 300
    C_list = np.zeros(shape = (2, n_ray))
    i = 0
    L_psi = np.linspace(-psi_max, psi_max, n_ray)
    L_tan_phi = np.zeros_like(L_psi)

    ### Building the new curve
    for psi in np.linspace(-psi_max, psi_max, n_ray):
                
        ### D is the intersection of the substrate with the current work ray
        func_xd = lambda xd : xd-Pos_buse[0] + (f_profile(xd) - H)*np.tan(theta + psi)
        xd = scipy.optimize.fsolve(func_xd, (Pos_buse[0] + H*np.tan(theta+psi))/2)[0] 
        yd = f_profile(xd)

        ### Calculating the angle phi between ray and profile for x = xd
        curr_slope = d_profile(xd)
        phi = - theta - psi + np.arctan(curr_slope)
        tan_phi = np.tan(phi)

        ### A is the intersection of the ray with the virtual gaussian curve representing the spray distribution

        func_xa = lambda xa : xa - np.tan(psi)*(h-G(xa))
        xa = scipy.optimize.fsolve(func_xa, 0)[0] 
        ya = G(xa)

        ### B is the intersection of the ray with the base of the virtual gaussian curve      
        xb, yb = dist_head_curve*np.tan(psi), 0

        ### We find the position of C by using the equation (in vector form) CD = DE(tan(phi)) * AB 
        DE_phi = effeciency(tan_phi)

        ### C is the new value of the profile along the current ray
        xc = xd + DE_phi * (xa - xb)
        yc = yd + DE_phi * (ya - yb)

        C_list[0][i] = xc
        C_list[1][i] = yc
        i += 1



    # Replace profile values with the added bump values

    f_C_list = scipy.interpolate.interp1d(C_list[0], C_list[1], kind = "cubic")
    def new_f_profile(x):
        x_min, x_max = C_list[0][0], C_list[0][-1]
        if x_min < x < x_max :
            y = f_C_list(x)
        else:
            y = f_profile(x)
        if y < 0 :
            return -y
        return y
    v_new_f_profile = np.vectorize(new_f_profile)
    new_profile = [profile[0], v_new_f_profile(profile[0])]

    return C_list

H = 15
theta_max = 0
Pos_buse = np.array([0, H])

L_x = np.linspace(-20, 20, 500)
L_y = np.zeros_like(L_x)
substrat = np.array([L_x, L_y])
profile = substrat
i_max = 20

for i in range(i_max):
    theta = theta_max#*(i-i_max/2)/ i_max 
    profile = rotation_lim(theta, Pos_buse, profile, max_reflexion_angle=np.pi/10)
    print(f"profile {i+1} done !")
    if i%5 == 0:
        plt.scatter(profile[0], profile[1], s = 2)


### Plotting the nozzle head position
plt.scatter(Pos_buse[0], Pos_buse[1])

### Plotting the max angle values of nozzle spray as well as center ray
plt.plot([Pos_buse[0], Pos_buse[0] + np.tan(theta_max) * Pos_buse[1]],[Pos_buse[1], 0], ":g")
plt.plot([Pos_buse[0], Pos_buse[0] + np.tan(theta_max + np.pi/10) * Pos_buse[1]],[Pos_buse[1], 0], ":y")
plt.plot([Pos_buse[0], Pos_buse[0] + np.tan(theta_max - np.pi/10) * Pos_buse[1]],[Pos_buse[1], 0], ":y")

plt.axis("equal")
plt.grid()
plt.show()




# X = [-2, 0, 3, 9, 12.3, 13.9, 23, 24, 55.2]
# Y = [2, 5, 0.3, -14, 23, 25, 26, 15, -0.3]

# X_hat, Y_hat = savgol_filter((X, Y), window_length=5, polyorder=3)

# plt.plot(X, Y)
# plt.plot(X_hat, Y_hat, ":r")
# plt.grid()
# plt.show()
