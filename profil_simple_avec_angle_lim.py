import numpy as np
import matplotlib.pyplot as plt
import intersect


def rotation_lim(theta, Pos_buse, profile = [], max_reflexion_angle = np.pi/6):
    """profile is a 2d list. profile[0] is the x step between each height value of the profile. profile[1] is a list 
    containing the height values of the profile"""


    if len(profile) == 0 :
        psi_max = 0.7
        sigma_p = 0.5#2*H*np.tan(psi_max)/5
        mu_p  = 0
        f = lambda t : 1/(sigma_p* np.sqrt(2*np.pi)) * np.exp(- (t-mu_p)**2/(2*sigma_p**2))
        L_x = np.linspace(-3, 3, 100)
        profile = [L_x, f(L_x)]



    ### Determining the intersection point of the primary ray of the spray head and the profile
    ray0 = np.array([[Pos_buse[0], Pos_buse[1]*np.tan(theta)+Pos_buse[0]], 
            [Pos_buse[1], 0]])
    xO, yO = intersect.intersection(profile[0], profile[1], ray0[0], ray0[1])

    xO = xO[0]
    yO = yO[0]


    ### Determining the sigma value for the virtual gaussian curve
    dist_head_curve = np.sqrt((Pos_buse[0] - xO)**2 + (Pos_buse[1] - yO)**2)
    psi_max = np.pi/10 # Max spray angle of the nozzle

    sigma = dist_head_curve*np.tan(psi_max)/5
    mu = 0
    G = lambda t : 1/(sigma* np.sqrt(2*np.pi)) * np.exp(- (t-mu)**2/(2*sigma**2))/5

    x_gauss = np.linspace(-5*sigma, 5*sigma, 100)
    y_gauss = G(x_gauss)
    Gauss_curve = np.array([x_gauss, y_gauss])

    slopes = np.diff(profile[1])

    def effeciency(tan_phi):
        k = 50
        tan_phi_max = np.tan(max_reflexion_angle)
        # tan_phi_max = 1.2
        if tan_phi < tan_phi_max:
            return (np.exp(-k*(tan_phi/tan_phi_max)**2) - np.exp(-k))/(1-np.exp(-k))
        else :
            print("ricochet")
            return 0

    n_ray = 100
    C_list = np.zeros(shape = (2, n_ray))
    i = 0

    ### Building the new curve
    for psi in np.linspace(-psi_max, psi_max, n_ray):
        real_ray = np.array([[Pos_buse[0], Pos_buse[1]*np.tan(psi+theta)+Pos_buse[0]], 
            [Pos_buse[1], 0]])
        
        virtual_ray = np.array([[0, dist_head_curve*np.tan(psi)], [dist_head_curve, 0]])
        
        ### D is the intersection of the substrate with the current work ray
        xd, yd = intersect.intersection(profile[0], profile[1], real_ray[0], real_ray[1])

        if xd.size == 0 :
            xd, yd = intersect.intersection(profile[0]*0.999, profile[1]*0.999, real_ray[0], real_ray[1])

        else : 
            xd = xd[0]
            yd = yd[0]

        ###Looking for the index in slpoes that corresponds to the slope in x = xd
        min_step_slope = 1
        i_slope = np.where((profile[0] > xd - min_step_slope) & (profile[0] < xd + min_step_slope))[0][0]
        curr_slope = slopes[i_slope]
        tan_phi = np.tan(theta + psi - np.arctan(curr_slope))
        
        ### A is the intersection of the ray with the virtual gaussian curve representing the spray distribution

        xa, ya = intersect.intersection(Gauss_curve[0], Gauss_curve[1], virtual_ray[0], virtual_ray[1])  
        
        xa = xa[0]
        ya = ya[0]

        ### B is the intersection of the ray with the base of the virtual gaussian curve      
        xb, yb = dist_head_curve*np.tan(psi), 0

        ### We find the position of C by using the equation (in vector form) CD = DE(tan(phi)) * AB 
        DE_phi = effeciency(tan_phi)

        xc = xd + DE_phi * (xa - xb)
        yc = yd + DE_phi * (ya - yb)

        C_list[0][i] = xc
        C_list[1][i] = yc
        i += 1

    ### Replace profile values with the added bump values
    min_step = 0.1
    x_min, x_max = C_list[0][0], C_list[0][-1]
    i_min = np.where((profile[0] > x_min - min_step) & (profile[0] < x_min + min_step))[0][0]
    i_max = np.where((profile[0] > x_max - min_step) & (profile[0] < x_max + min_step))[0][0]

    new_profile_x = np.concatenate(( np.concatenate((profile[0][:i_min], C_list[0])) , profile[0][i_max:]))
    new_profile_y = np.concatenate(( np.concatenate((profile[1][:i_min], C_list[1])) , profile[1][i_max:]))

    new_profile = [new_profile_x, new_profile_y]

    return new_profile

H = 5
theta_max = 0
pas = 0.001
Pos_buse = np.array([0, H])

f= lambda x : -1/50 * (x-8)**2 + 3
L_x = np.linspace(-5, 5, 300)
L_y = f(L_x)

L_y1 = np.zeros_like(L_x)
substrat = np.array([L_x, L_y1])


profile = substrat

for i in range(10):
    theta = theta_max 
    profile = rotation_lim(theta, Pos_buse, profile, max_reflexion_angle=np.pi/3)
    if i%2 == 0:
        plt.plot(profile[0], profile[1])


plt.scatter(Pos_buse[0], Pos_buse[1])
plt.plot([Pos_buse[0], Pos_buse[0] + np.tan(theta) * Pos_buse[1]],[Pos_buse[1], 0], ":g")
plt.plot([Pos_buse[0], Pos_buse[0] + np.tan(theta + np.pi/10) * Pos_buse[1]],[Pos_buse[1], 0], ":y")
plt.plot([Pos_buse[0], Pos_buse[0] + np.tan(theta - np.pi/10) * Pos_buse[1]],[Pos_buse[1], 0], ":y")
plt.axis("equal")
plt.grid()
plt.show()

