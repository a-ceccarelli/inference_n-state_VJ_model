import numpy as np
import scipy
import scipy.stats
import copy
import random
import matplotlib.pyplot as plt
from scipy.linalg import null_space

"""
This file functions.py contains the n-state model simulation function 'n_state_model'

It contains the up-to-one-switch approximation function 'approx_pdf_up_to_1_switch'
which should work for any model network

It contains the up-to-two-switch approximation function 'approx_pdf_up_to_2_switch'
which should work for model networks in which states have distinct switching rates
lambda and distinct velocities

It contains the track PDF up-to-one-switch approximation function
'approx_pdf_track_up_to_1_switch' which should work for any model network

If the up-to-one-switch approximations give numerical errors for velocities or rates
of distinct states one could use the approximation in which they are equal, modifying
the tolerance eps, now set to 1e-14
"""


def n_state_model(delta_t, T, V, Lambda, P, sigma, x0=0.0, t0=0.0, state='?', seed=1223,
                  plot=False, plot_png_title=None, plot_legend=False, plot_ylabel=None):
    """
    Simulates axonal transport
    delta_t = time step (s) 8nm / 800nm/s = 0.01s
    T = total time (s)
    
    #all velocities (with sign)   
    V = [v_1, v_2, ..., v_n] # here indexed 0,1,...,n-1
    
    # rates time spent on state i
    Lambda = [lambda_1, lambda_2, ..., lambda_n] = (lambda_i) # here indexed 0,1,...,n-1
    
    # transition matrix: p_ij probability of switching from i to j
    P = (p_ij)            
    
    sigma #noise std
    
    x0 = initial position
    state = initial state known or '?'
    seed = select random seed
    """
    # set a random seed
    np.random.seed(seed)
    random.seed(seed)
    
    if delta_t<=0 or T<delta_t:
        print('Error defining time')
        return None
    
    n = Lambda.shape[0]
    
    if np.any(Lambda <= np.zeros(n)):
        print('Error defining lambdas')
        return None
    
    if n!=P.shape[0] or n!=P.shape[1] or n!=V.shape[0]:
        print('Error with dimensions')
        return None
    
    if np.any(np.diag(P) != np.zeros(n)):
        print('Error defining transition matrix P diag')
        return None
        
    if np.any(np.sum(P, axis=1) != np.ones(n)):
        print('Error defining transition matrix P sum')
        return None
    
    if np.any(sigma < 0):
        print('Error defining noise parameter sigma')
        return None
        
    t = [t0]
    x = [x0]
    curr_dt = 0
    curr_dx = 0

    #Building Q transpose
    Q = np.zeros((n,n))
    for i in range(n):
        for j in range(0,i):
            Q[i,j] = Lambda[i]*P[i,j]
        Q[i,i] = -Lambda[i]
        for j in range(i+1,n):
            Q[i,j] = Lambda[i]*P[i,j]
    Qt = Q.T
    
    #Computing w s.t. Q^T w = 0 and then P(st)
    W = null_space(Qt)
    P_st = W / np.sum(W)
    
    #Choosing an initial state
    if state=='?': #state -1 is unknown
        state = random.choices(np.arange(n), weights=P_st, k=1)[0]
        
    state_save = []
        
    while t[-1]<T: # T is in seconds again
        curr_dt = np.random.exponential(scale=1/Lambda[state])
        curr_dx = curr_dt*V[state]
        t.append(t[-1]+curr_dt)
        x.append(x[-1]+curr_dx)
        state_save.append(state)
        state = random.choices(np.arange(n), weights=P[state, :], k=1)[0]
    
    state_save.append(state)
    
    #CREATE delta_t=0.3 approx
    t_points = np.linspace(0, T, int(T/delta_t)+1) #e.g. 0, 0.3, 0.6, ..., 60
    #print(x,t)
    # generate error points
    x_points = np.zeros(t_points.shape)
    y_points = np.random.normal(loc=0, scale=sigma,
                                size=t_points.shape)
    switches = np.zeros(t_points.shape)
    all_states_save = [[] for i in range(int(T/delta_t)+1)]
    
    j = 0
    p = np.polyfit(t[j:j+2], x[j:j+2], 1)
    for i in range(0,int(T/delta_t)+1):
        todo = True
        while todo:
            all_states_save[i].append(state_save[j])
            if t_points[i]>=t[j] and t_points[i]<=t[j+1]:
                x_points[i] += np.polyval(p, t_points[i])
                y_points[i] += x_points[i]
                todo = False
                
                if t_points[i]==t[j]:
                    if i!= 0:
                        switches[i] += 1
            else:
                j+=1
                switches[i] +=1
                p = np.polyfit(t[j:j+2], x[j:j+2], 1)

    if plot:
        plt.figure(figsize=(1.7,2))
        plt.plot(y_points, t_points, '*', color='magenta', label=r'$(y_j, t_j)$')
        plt.plot(x, t, color='purple', label=r'$(x(t), t)$')
        plt.ylim([-0.1, T+0.1])
        if plot_legend:
            plt.legend()    
        plt.gca().invert_yaxis()
        if plot_ylabel is not None:
            plt.ylabel(plot_ylabel)
        plt.savefig(plot_png_title, format="png", dpi=1200, bbox_inches="tight") 
                
    return x_points, y_points, t_points, switches[1:], all_states_save[1:]



def approx_pdf_up_to_1_switch(V, Lambda, P, sigma, delta_t, delta_y, eps=1e-14):
    #checks
    if delta_t<=0:
        print('Error defining time')
        return None
    
    n = Lambda.shape[0]
    
    if np.any(Lambda <= np.zeros(n)):
        print('Error defining lambdas')
        return None
    
    if n!=P.shape[0] or n!=P.shape[1] or n!=V.shape[0]:
        print('Error with dimensions')
        return None
    
    if np.any(np.diag(P) != np.zeros(n)):
        print('Error defining transition matrix P diag')
        return None
        
    if np.any(np.sum(P, axis=1) != np.ones(n)):
        print('Error defining transition matrix P sum')
        return None
    
    if np.any(sigma < 0):
        print('Error defining noise parameter sigma')
        return None
    
    #Building Q transpose
    Q = np.zeros((n,n))
    for i in range(n):
        for j in range(0,i):
            Q[i,j] = Lambda[i]*P[i,j]
        Q[i,i] = -Lambda[i]
        for j in range(i+1,n):
            Q[i,j] = Lambda[i]*P[i,j]
    Qt = Q.T
    
    #Computing w s.t. Q^T w = 0 and then P(st)
    W = null_space(Qt)
    P_st = W / np.sum(W)
    
    P_sw_ge1_giv_st = [scipy.stats.expon.cdf(delta_t, loc=0, scale=1/Lambda[i])
                       for i in range(n)]

    P_sw_0_giv_st = [1-scipy.stats.expon.cdf(delta_t, loc=0, scale=1/Lambda[i])
                     for i in range(n)]

    def P_dy_giv_sw_0_st(delta_y, st):
        return scipy.stats.norm.pdf(delta_y, loc=V[st]*delta_t,
                                    scale=np.sqrt(2)*sigma)

    def P_dy_giv_sw_1_st1_st2(delta_y, st1, st2):
    
        def h(t):
            return V[st1]*t + V[st2]*(delta_t-t)
        h_0 = h(0)
        h_delta_t = h(delta_t)
        
        a = min([h_0, h_delta_t])
        b = max([h_0, h_delta_t])

        if np.abs(V[st1]-V[st2])<=eps: #or maybe check for close enough
            return scipy.stats.norm.pdf(delta_y, loc=V[st1]*delta_t,
                                        scale=np.sqrt(2)*sigma)

        if np.abs(Lambda[st1]-Lambda[st2])<=eps:
            res = scipy.stats.norm.cdf(delta_y-a, loc=0, scale=np.sqrt(2)*sigma)
            res -= scipy.stats.norm.cdf(delta_y-b, loc=0, scale=np.sqrt(2)*sigma)
            return res/(b-a)

        #otherwise, all the other cases

        H = 1/(V[st1]-V[st2])
        s = np.sign(V[st1]-V[st2])

        c = 1/(4*(sigma**2))
        L = (-Lambda[st1]+Lambda[st2])

        def f():
            return (2*c*delta_y) + (L*H)
        f = f()

        def g():
            return (-c*(delta_y**2)) + (L*s*H*np.abs(V[st2])*delta_t)
        g = g()

        def k():
            return s * H * (L/2) * np.exp(((f**2)/(4*c)) + g) / (np.exp(L*delta_t)-1)
        k = k()

        res = k * (scipy.special.erf((np.subtract(2*c*b,f))/(2*np.sqrt(c))) -
                scipy.special.erf((np.subtract(2*c*a,f))/(2*np.sqrt(c))))

        return res
        
    def P_delta_y_giv_st(delta_y, st):
        I = np.array([i for i in np.arange(st)] + [i for i in np.arange(st+1, n)])
        s1 = P_dy_giv_sw_0_st(delta_y, st)*P_sw_0_giv_st[st] 
        s2 = sum([P_dy_giv_sw_1_st1_st2(delta_y, st, st2)*P[st,st2]*P_sw_ge1_giv_st[st] for st2 in I])
        return (s1 + s2)
    
    approx_pdf = sum([P_delta_y_giv_st(delta_y, st)*P_st[st] for st in np.arange(n)])
    
    return approx_pdf



def approx_pdf_up_to_2_switch(V, Lambda, P, sigma, delta_t, delta_y, eps=1e-14):
    #checks
    if delta_t<=0:
        print('Error defining time')
        return None
    
    n = Lambda.shape[0]
    
    if np.any(Lambda <= np.zeros(n)):
        print('Error defining lambdas')
        return None
    
    if n!=P.shape[0] or n!=P.shape[1] or n!=V.shape[0]:
        print('Error with dimensions')
        return None
    
    if np.any(np.diag(P) != np.zeros(n)):
        print('Error defining transition matrix P diag')
        return None
        
    if np.any(np.sum(P, axis=1) != np.ones(n)):
        print('Error defining transition matrix P sum')
        return None
    
    if np.any(sigma < 0):
        print('Error defining noise parameter sigma')
        return None
    
    #Building Q transpose
    Q = np.zeros((n,n))
    for i in range(n):
        for j in range(0,i):
            Q[i,j] = Lambda[i]*P[i,j]
        Q[i,i] = -Lambda[i]
        for j in range(i+1,n):
            Q[i,j] = Lambda[i]*P[i,j]
    Qt = Q.T
    
    #Computing w s.t. Q^T w = 0 and then P(st)
    W = null_space(Qt)
    P_st = W / np.sum(W)

    P_sw_0_giv_st = np.array([1-scipy.stats.expon.cdf(delta_t, loc=0, scale=1/Lambda[i])
                              for i in range(n)])


    def P_sw_1_giv_st1_st2(st1, st2):
        if st1==st2:
            print('Error?')
            return 0
        if Lambda[st1] == Lambda[st2]:
            return Lambda[st1]*np.exp(-Lambda[st1]*delta_t)*delta_t
        M1 = Lambda[st1] * np.exp(-Lambda[st2]*delta_t)/(-Lambda[st1]*Lambda[st2])
        M2 = np.exp((-Lambda[st1]*Lambda[st2])*delta_t) - 1
        return M1 * M2


    def P_sw_ge2_giv_st1_st2(st1, st2):
        return 1 - P_sw_0_giv_st[st1] - P_sw_1_giv_st1_st2(st1, st2)


    def P_dy_giv_sw_0_st(delta_y, st):
        return scipy.stats.norm.pdf(delta_y, loc=V[st]*delta_t,
                                    scale=np.sqrt(2)*sigma)

    def P_dy_giv_sw_1_st1_st2(delta_y, st1, st2):

        if np.abs(V[st1]-V[st2])<=eps: #or maybe check for close enough
            return scipy.stats.norm.pdf(delta_y, loc=V[st1]*delta_t,
                                        scale=np.sqrt(2)*sigma)

        if np.abs(Lambda[st1]-Lambda[st2])<=eps:
            res = scipy.stats.norm.cdf(delta_y-a, loc=0, scale=np.sqrt(2)*sigma)
            res -= scipy.stats.norm.cdf(delta_y-b, loc=0, scale=np.sqrt(2)*sigma)
            return res/(b-a)

        #otherwise, all the other cases

        def h(t):
            return V[st1]*t + V[st2]*(delta_t-t)
        h_0 = h(0)
        h_delta_t = h(delta_t)

        a = min([h_0, h_delta_t])
        b = max([h_0, h_delta_t])


        H = 1/(V[st1]-V[st2])
        s = np.sign(V[st1]-V[st2])

        c = 1/(4*(sigma**2))
        L = (-Lambda[st1]+Lambda[st2])

        def f():
            return (2*c*delta_y) + (L*H)
        f = f()

        def g():
            return (-c*(delta_y**2)) + (L*s*H*np.abs(V[st2])*delta_t)
        g = g()

        def k():
            return s * H * (L/2) * np.exp(((f**2)/(4*c)) + g) / (np.exp(L*delta_t)-1)
        k = k()

        res = k * (scipy.special.erf((np.subtract(2*c*b,f))/(2*np.sqrt(c))) -
                scipy.special.erf((np.subtract(2*c*a,f))/(2*np.sqrt(c))))

        return res
    

    def P_dy_giv_sw_2_st1_st2_st3(delta_Y, st1, st2, st3):
        #would need to exclude same state or same lambda or same V first
        list_vec_ret = []
        V_ord = np.sort(np.array([V[st1], V[st2], V[st3]])) #V_ord = [V_min, V_int, V_max]
        #print(V_ord)
        
        #If st1 and st3 are the same it is a different case
        if st1==st3:
            c = 1/(4*(sigma**2))
            L12 = Lambda[st1] - Lambda[st2]
            a = V_ord[0] * delta_t
            b = V_ord[2] * delta_t
            for delta_y in delta_Y:
                f = 2 * c * delta_y + L12 / (V[st2] - V[st1]) 
                g =  - c * (delta_y**2) + (L12 / np.abs(V[st2] - V[st1])) * (np.abs(V[st1])
                                                                        * delta_t)
                
                k = ((L12**2) * (sigma**2) * np.exp(g) /
                    (np.sqrt(np.pi) * np.abs(V[st2] - V[st1]) * (V[st1] - V[st2]) 
                    * (np.exp(L12 * delta_t) - 1 - L12 * delta_t)))
            
                res = k * (np.sqrt(np.pi) * np.exp(f**2/(4*c)) 
                        * (2*c*(-V[st2]*delta_t)+f) * 
                        (scipy.special.erf((2*c*b-f)/(2*np.sqrt(c))) -
                            scipy.special.erf((2*c*a-f)/(2*np.sqrt(c))))
                        - 2*np.sqrt(c)*(np.exp(b*(f-c*b))-np.exp(a*(f-c*a))))
                list_vec_ret.append(copy.deepcopy(res))
                
            return np.array(list_vec_ret)
        
        
        #If lambda and V all distinct
        dict_for_ord = {V[st1]:st1, V[st2]:st2, V[st3]:st3}
        Lambda_ord = np.array([Lambda[dict_for_ord[V_ord[0]]],
                            Lambda[dict_for_ord[V_ord[1]]],
                            Lambda[dict_for_ord[V_ord[2]]]])
        #we rearrange in the most convenient order 
        V_conv = {1:V_ord[1], 2:V_ord[2], 3:V_ord[0]}
        Lambda_conv = {1:Lambda_ord[1], 2:Lambda_ord[2], 3:Lambda_ord[0]}
        
        c = 1/(4*(sigma**2))
        L12 = Lambda_conv[1] - Lambda_conv[2]
        L23 = Lambda_conv[2] - Lambda_conv[3]
        L31 = Lambda_conv[3] - Lambda_conv[1]
        k123 = -L12*L23*L31 * np.exp(delta_t*(V_conv[2]*(Lambda_conv[1]+Lambda_conv[2])
                                            -V_conv[3]*(Lambda_conv[1]+Lambda_conv[3]))
                                    /(V_conv[2]-V_conv[3]))
        D12 = L12 * np.exp((Lambda_conv[1]+Lambda_conv[2])*delta_t)
        D23 = L23 * np.exp((Lambda_conv[2]+Lambda_conv[3])*delta_t)
        D31 = L31 * np.exp((Lambda_conv[3]+Lambda_conv[1])*delta_t)
        D = D12 + D23 + D31
        nu123 = V_conv[1]*L23 + V_conv[2]*L31 + V_conv[3]*L12
        
        for delta_y in delta_Y:
            #J_0
            f0 = 2 * c * delta_y + L23 / (V_conv[3] - V_conv[2]) 
            g0 =  - c * (delta_y**2) 
            k0 = (k123 * np.sign(V_conv[3]-V_conv[2]) * np.exp(((f0**2)/(4*c)) + g0)
                / (2 * D * nu123))
            J0 = k0 * (scipy.special.erf((2*c*(V_ord[2]*delta_t)-f0)/(2*np.sqrt(c)))
                    - scipy.special.erf((2*c*(V_ord[0]*delta_t)-f0)/(2*np.sqrt(c))))
            
            #J_b1
            common_term_b1 = nu123 / ((V_conv[2] - V_conv[3])*(V_conv[1] - V_conv[3]))
            fb1 = f0 + common_term_b1
            gb1 = g0 - (common_term_b1 * V_conv[3] * delta_t)
            kb1 = (k123 * np.sign(V_conv[2]-V_conv[3]) * np.exp(((fb1**2)/(4*c)) + gb1)
                / (2 * D * nu123))
            Jb1 = kb1 * (scipy.special.erf((2*c*(V_ord[1]*delta_t)-fb1)/(2*np.sqrt(c)))
                        - scipy.special.erf((2*c*(V_ord[0]*delta_t)-fb1)/(2*np.sqrt(c))))
            
            #J_a1
            common_term_a1 = nu123 / ((V_conv[2] - V_conv[3])*(V_conv[1] - V_conv[2])) 
            fa1 = f0 + common_term_a1
            ga1 = g0 - (common_term_a1 * V_conv[2] * delta_t)
            ka1 = (k123 * np.sign(V_conv[2]-V_conv[3]) * np.exp(((fa1**2)/(4*c)) + ga1)
                / (2 * D * nu123))
            Ja1 = ka1 * (scipy.special.erf((2*c*(V_ord[2]*delta_t)-fa1)/(2*np.sqrt(c)))
                        - scipy.special.erf((2*c*(V_ord[1]*delta_t)-fa1)/(2*np.sqrt(c))))
            
            list_vec_ret.append(J0 + Ja1 + Jb1)
        
        return np.array(list_vec_ret)

        
    def P_delta_y_giv_st(delta_y, st):
        I = np.array([i for i in np.arange(st)] + [i for i in np.arange(st+1, n)])
        s1 = P_dy_giv_sw_0_st(delta_y, st)*P_sw_0_giv_st[st] 
        s2 = sum([P_dy_giv_sw_1_st1_st2(delta_y, st, st2)*P[st,st2]
                  *P_sw_1_giv_st1_st2(st, st2) for st2 in I])
        s3 = 0
        for st2 in I:
            J = np.array([i for i in np.arange(st2)] + [i for i in np.arange(st2+1, n)])
            s3 += sum([P_dy_giv_sw_2_st1_st2_st3(delta_y, st, st2, st3)
                       *P[st,st2]*P[st2,st3]*P_sw_ge2_giv_st1_st2(st,st2) for st3 in J])
        return (s1 + s2 + s3)
    
    approx_pdf = sum([P_delta_y_giv_st(delta_y, st)*P_st[st] for st in np.arange(n)])
    
    return approx_pdf



def approx_pdf_track_up_to_1_switch(V, Lambda, P, sigma, delta_t, delta_y_track, eps=1e-14):
    #checks
    if delta_t<=0:
        print('Error defining time')
        return None
    
    n = Lambda.shape[0]
    
    if np.any(Lambda <= np.zeros(n)):
        print('Error defining lambdas')
        return None
    
    if n!=P.shape[0] or n!=P.shape[1] or n!=V.shape[0]:
        print('Error with dimensions')
        return None
    
    if np.any(np.diag(P) != np.zeros(n)):
        print('Error defining transition matrix P diag')
        return None
        
    if np.any(np.sum(P, axis=1) != np.ones(n)):
        print('Error defining transition matrix P sum')
        return None
    
    if np.any(sigma < 0):
        print('Error defining noise parameter sigma')
        return None
    
    #Building Q transpose
    Q = np.zeros((n,n))
    for i in range(n):
        for j in range(0,i):
            Q[i,j] = Lambda[i]*P[i,j]
        Q[i,i] = -Lambda[i]
        for j in range(i+1,n):
            Q[i,j] = Lambda[i]*P[i,j]
    Qt = Q.T
    
    #Computing w s.t. Q^T w = 0 and then P(st)
    W = null_space(Qt)
    P_st = W / np.sum(W)
    
    P_sw_ge1_giv_st = [scipy.stats.expon.cdf(delta_t, loc=0, scale=1/Lambda[i])
                       for i in range(n)]

    P_sw_0_giv_st = [1-scipy.stats.expon.cdf(delta_t, loc=0, scale=1/Lambda[i])
                     for i in range(n)]

    def P_dy_giv_sw_0_st(delta_y, st):
        return scipy.stats.norm.pdf(delta_y, loc=V[st]*delta_t,
                                    scale=np.sqrt(2)*sigma)

    def P_dy_giv_sw_1_st1_st2(delta_y, st1, st2):
        if np.abs(V[st1]-V[st2])<=eps: #or maybe check for close enough
            return scipy.stats.norm.pdf(delta_y, loc=V[st1]*delta_t,
                                        scale=np.sqrt(2)*sigma)

        if  np.abs(Lambda[st1]-Lambda[st2])<=eps:
            res = scipy.stats.norm.cdf(delta_y-a, loc=0, scale=np.sqrt(2)*sigma)
            res -= scipy.stats.norm.cdf(delta_y-b, loc=0, scale=np.sqrt(2)*sigma)
            return res/(b-a)

        #otherwise, all the other cases

        def h(t):
            return V[st1]*t + V[st2]*(delta_t-t)
        h_0 = h(0)
        h_delta_t = h(delta_t)

        a = min([h_0, h_delta_t])
        b = max([h_0, h_delta_t])


        H = 1/(V[st1]-V[st2])
        s = np.sign(V[st1]-V[st2])

        c = 1/(4*(sigma**2))
        L = (-Lambda[st1]+Lambda[st2])

        def f():
            return (2*c*delta_y) + (L*H)
        f = f()

        def g():
            return (-c*(delta_y**2)) + (L*s*H*np.abs(V[st2])*delta_t)
        g = g()

        def k():
            return s * H * (L/2) * np.exp(((f**2)/(4*c)) + g) / (np.exp(L*delta_t)-1)
        k = k()

        res = k * (scipy.special.erf((np.subtract(2*c*b,f))/(2*np.sqrt(c))) -
                scipy.special.erf((np.subtract(2*c*a,f))/(2*np.sqrt(c))))

        return res
        
    def P_delta_y_giv_st(delta_y, st):
        I = np.array([i for i in np.arange(st)] + [i for i in np.arange(st+1, n)])
        s1 = P_dy_giv_sw_0_st(delta_y, st)*P_sw_0_giv_st[st] 
        s2 = sum([P_dy_giv_sw_1_st1_st2(delta_y, st, st2)*P[st,st2]*P_sw_ge1_giv_st[st] for st2 in I])
        #print(s1,s2)
        return (s1 + s2)
    
    
    #Now we include the modifications useful for the P(track)
    N = delta_y_track.shape[0]
    
    #list of arrays: P(delta_y_i | st_i^{(1)}) [i][st]
    P_delta_y_giv_st_vec = np.array([[P_delta_y_giv_st(delta_y_track[i], st) 
                                      for st in np.arange(n)] for i in np.arange(N)])
        
    #list of arrays: P(st_i | delta_y_i-1,...,delta_y_1) [i][st]
    P_st_i_giv_all_prev_delta_y_vec = [np.array(n) for i in np.arange(N)]
    P_st_i_giv_all_prev_delta_y_vec[0] = P_st.copy()
    
    #array (N)
    #print((P_delta_y_giv_st_vec).shape) - it is (2,3)
    #[P(delta_y_1), P(delta_y_2 | delta_y_1), P(delta_y_3 | delta_y_2, delta_y_1), ...]
    approx_pdf_vec_track = np.zeros(N) 
    approx_pdf_vec_track[0] = sum([P_delta_y_giv_st_vec[0,st]*P_st[st]
                                   for st in np.arange(n)])
    
    def P_delta_y_i_giv_all_prev_delta_y(i):
        ret = sum([P_delta_y_giv_st_vec[i][st] 
                   * P_st_i_giv_all_prev_delta_y_vec[i][st]
                   for st in np.arange(n)])
        return ret
    
    
    def P_delta_y_i_and_st_ip1_giv_st_i(i, st_ip1, st_i): # order-one approx
        if st_ip1 == st_i:
            return P_dy_giv_sw_0_st(delta_y_track[i], st_i)*P_sw_0_giv_st[st_i] 
        return P_dy_giv_sw_1_st1_st2(delta_y_track[i], st_i, st_ip1)*P_sw_ge1_giv_st[st_i]*P[st_i,st_ip1]
    
    
    def P_st_i_giv_all_prev_delta_y(st, i):
        ret = sum([P_delta_y_i_and_st_ip1_giv_st_i(i-1, st, st_im1)  
                   * P_st_i_giv_all_prev_delta_y_vec[i-1][st_im1]
                   / approx_pdf_vec_track[i-1]
                   for st_im1 in np.arange(n)])
        return ret

    #Now I have defined the functions, I need to use them
    
    for i in range(1, N):
        P_st_i_giv_all_prev_delta_y_vec[i] = np.array([P_st_i_giv_all_prev_delta_y(st, i)
                                                       for st in np.arange(n)])
        approx_pdf_vec_track[i] = P_delta_y_i_giv_all_prev_delta_y(i)
    #CHECK INDECES!
    
    return approx_pdf_vec_track #[P(delta_y_1), P(delta_y_2 | delta_y_1), P(delta_y_3 | delta_y_2, delta_y_1), ... ]