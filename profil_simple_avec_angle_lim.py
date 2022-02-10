import enum
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize
from scipy.signal import savgol_filter

def rotation_lim(theta, Pos_buse, profile):
    """profile is a 2d list. profile[0] is the x values of each point of the profile. profile[1] is a list 
    containing the height values of the profile for each x"""


    ### Creating interpolated function of the profile
    f_profile = scipy.interpolate.interp1d(profile[0], profile[1], kind = "cubic", fill_value='extrapolate')

    ### Max spray angle of the nozzle
    psi_max = np.pi/10 


    ### Determining the x span hit by the nozzle spray
    func_xmin = lambda x : x-Pos_buse[0] + (f_profile(x) - H)*np.tan(theta-1*psi_max)
    x_min = scipy.optimize.fsolve(func_xmin, (Pos_buse[0] + H*np.tan(theta-1*psi_max))/2)[0] 

    func_xmax = lambda x : x-Pos_buse[0] + (f_profile(x) - H)*np.tan(theta+1*psi_max)
    x_max = scipy.optimize.fsolve(func_xmax, (Pos_buse[0] + H*np.tan(theta+1*psi_max))/2)[0] 


    ### Creating a work profile that spans from x_min to x_max and has about the same dx step in between two values of x than the input profile
    N_values = int(np.floor(len(profile[0])/(profile[0][-1] - profile[0][1]) * (x_max - x_min)+0.5))
    if N_values < 20:
        N_values = 20

    L_x = np.linspace(x_min, x_max, N_values)
    work_profile_y = np.zeros_like(L_x)

    for i in range(len(L_x)):
        work_profile_y[i] = f_profile(L_x[i])
    work_profile = np.array([L_x, work_profile_y])


    ### Smoothing work profile
    w_length = len(work_profile[0])//2
    if w_length % 2 == 0:
        w_length += 1
    
    work_profile[0], work_profile[1] = savgol_filter((work_profile[0], work_profile[1]), window_length=w_length, polyorder=4)
    f_work_profile = scipy.interpolate.interp1d(work_profile[0], work_profile[1], kind = "cubic", fill_value='extrapolate')

    ### Determining the intersection point of the primary ray of the nozzle and the work profile
    func_x0 = lambda x0 : x0-Pos_buse[0] + (f_work_profile(x0) - H)*np.tan(theta)
    x0 = scipy.optimize.fsolve(func_x0, (Pos_buse[0] + H*np.tan(theta))/2)[0] 
    y0 = f_work_profile(x0)

    ### Determining the sigma value for the virtual gaussian curve
    dist_head_curve = np.sqrt((Pos_buse[0] - x0)**2 + (Pos_buse[1] - y0)**2)
    h = dist_head_curve
    sigma = h*np.tan(psi_max)/5
    mu = 0
    G = lambda t : np.exp(- (t-mu)**2/(2*sigma**2))/(sigma* np.sqrt(2*np.pi))

    ### Calculating the slope values of the profile
    slopes = []
    for i in range(len(work_profile[0])-1):
        xi = work_profile[0][i]
        xi1 = work_profile[0][i+1]
        yi = work_profile[1][i]
        yi1 = work_profile[1][i+1]
        if xi-xi1 != 0:
            slope =  (yi1 - yi)/(xi1 - xi)
            slopes.append(np.floor(slope*1000)/1000)
    slopes.append(slope)

    d_profile_hat = [[],[]]
    d_profile_hat[0], d_profile_hat[1] = savgol_filter((work_profile[0], slopes), window_length=w_length, polyorder=5) #(work_profile[0], slopes)
    d_profile = scipy.interpolate.interp1d(d_profile_hat[0], d_profile_hat[1], kind = "cubic", fill_value='extrapolate')


    def effeciency(tan_phi):
        k = 4.5
        tan_phi_max = 1.7
        if tan_phi**2 < tan_phi_max**2:
            return (np.exp(-k*(tan_phi/tan_phi_max)**2) - np.exp(-k))/(1-np.exp(-k))
        else :
            # print("ricochet")
            return 0  #(np.exp(-k*(tan_phi/tan_phi_max)**2) - np.exp(-k))/(1-np.exp(-k))

    n_ray = 201
    C_list = np.zeros(shape = (2, n_ray))
    i = 0

    ### Building the new curve
    for psi in np.linspace(-psi_max, psi_max, n_ray):
    
        ### D is the intersection of the substrate with the current work ray
        func_xd = lambda xd : xd-Pos_buse[0] + (f_work_profile(xd) - H)*np.tan(theta + psi)
        xd = scipy.optimize.fsolve(func_xd, (Pos_buse[0] + H*np.tan(theta+psi))/2)[0] # This line sometimes raises a RuntimeWarning
        yd = f_work_profile(xd)

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



    ### Replace profile values with the added bump values
    f_C_list = scipy.interpolate.interp1d(C_list[0], C_list[1], kind = "cubic")

    def new_f_profile(x):
        x_min, x_max = C_list[0][0], C_list[0][-1]
        if x_min < x < x_max :
            y = f_C_list(x)
            if y < f_profile(x):
                # print("I'm here")
                y = f_profile(x)
        else:
            y = f_profile(x)
        if y < 0 :
            return y
        return y

    v_new_f_profile = np.vectorize(new_f_profile)
    new_profile = [profile[0], v_new_f_profile(profile[0])]
   
    return new_profile


def f_substrat(x):
    # if (x)**2/9 > 5:
    #     return 5
    # else:
    #     return (x)**2/9

    return 0 #+ 0.8*np.sin(2*np.pi*x/10)

def f_buse(x):
    if (x)**2/9 > 2:
        return 0#-(x)**2/10 + 6#5
    else:
        return 0#-(x)**2/9 + 6 #-(x)**2/10 + 6       

def f_theta(x_buse):
    return np.cos(x_buse)

def make_substrat():
    L_x = np.linspace(-40, 40, 1001)
    L_y = np.zeros_like(L_x)
            
    for i, x in enumerate(L_x):
        L_y[i] = f_substrat(x)

    plt.plot(L_x, L_y, "r")
    plt.axis("equal")
    plt.show()
    return np.array([L_x, L_y])



H = 20
theta_max = 0
Pos_buse = np.array([0, H])
substrat = make_substrat()
profile = substrat
i_max = 11

profiles = []
n_passes = 4
L_Pos_buse_pass = []

for n in range(n_passes):
    L_Pos_buse = [[],[]]
    f_x =  lambda i : (-10 + 20*i/(i_max-1))*(-1)**n
    f_H = lambda x_buse : H  + f_buse(x_buse) + n*0.3
    f_theta = lambda i : theta_max * np.cos(np.pi * i/(i_max-1))*(-1)**n

    for i in range(i_max):
        x_buse = f_x(i)
        theta = f_theta(i)
        Pos_buse = [x_buse, f_H(x_buse)]
        L_Pos_buse[0].append(Pos_buse[0])
        L_Pos_buse[1].append(Pos_buse[1])

        profile = rotation_lim(theta, Pos_buse, profile)
    # plt.plot([Pos_buse[0], Pos_buse[0] + np.tan(theta) * Pos_buse[1]],[Pos_buse[1], 0], ":g")
    print(f"Pass {n+1} done")
    new_profile = np.array([profile[0], profile[1]])
    profiles.append(new_profile)
    L_Pos_buse_pass.append(L_Pos_buse)

# for n in range(n_passes):

#     f_x =  lambda i : (-5 + 10*i/(i_max-1))*(-1)**n
#     f_y = lambda n : H + n*0.7
#     H = f_y(n)
#     for i in range(i_max):
#         theta = theta_max #-2*theta_max *(i-i_max/2)/ i_max 
#         Pos_buse = [f_x(i), H]

#         profile = rotation_lim(theta, Pos_buse, profile)
#         print(f"profile {i+1} done !")

#     new_profile = np.array([profile[0], profile[1]])
#     profiles.append(new_profile)

for i, profile in enumerate(profiles):
    plt.plot(profile[0], profile[1], label = f"pass number{i+1}")

for i, L_Pos_buse in enumerate(L_Pos_buse_pass):
    plt.scatter(L_Pos_buse[0], L_Pos_buse[1], s = 1)

### Plotting the nozzle head position
plt.scatter(Pos_buse[0], Pos_buse[1])

### Plotting the max angle values of nozzle spray as well as center ray
# plt.plot([Pos_buse[0], Pos_buse[0] + np.tan(theta_max) * Pos_buse[1]],[Pos_buse[1], 0], ":g")
# plt.plot([Pos_buse[0], Pos_buse[0] + np.tan(theta_max + np.pi/10) * Pos_buse[1]],[Pos_buse[1], 0], ":y")
# plt.plot([Pos_buse[0], Pos_buse[0] + np.tan(theta_max - np.pi/10) * Pos_buse[1]],[Pos_buse[1], 0], ":y")

### Plotting substrate surface
plt.plot(substrat[0], substrat[1], "r")

plt.legend()
plt.axis("equal")
plt.grid()
plt.show()
