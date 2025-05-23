import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import copy
import random
import pandas as pd
import scipy
from scipy.stats import norm, expon
from scipy.linalg import null_space
import math
import time

import os
cwd = os.getcwd()

def approx_pdf_track_up_to_1_switch(V, Lambda, P, sigma, delta_t, delta_y_track, eps=1e-14):
    
    n = Lambda.shape[0]
    
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
    
    if isinstance(delta_y_track, list):

        approx_pdf_vec_track = list()

        for delta_y_track_el in delta_y_track:
            #Now we include the modifications useful for the P(track)
            N = delta_y_track_el.shape[0]
            
            #list of arrays: P(delta_y_i | st_i^{(1)}) [i][st]
            P_delta_y_giv_st_vec = np.array([[P_delta_y_giv_st(delta_y_track_el[i], st) 
                                            for st in np.arange(n)] for i in np.arange(N)])
                
            #list of arrays: P(st_i | delta_y_i-1,...,delta_y_1) [i][st]
            P_st_i_giv_all_prev_delta_y_vec = [np.array(n) for i in np.arange(N)]
            P_st_i_giv_all_prev_delta_y_vec[0] = P_st.copy()
            
            #array (N)
            #print((P_delta_y_giv_st_vec).shape) - it is (2,3)
            #[P(delta_y_1), P(delta_y_2 | delta_y_1), P(delta_y_3 | delta_y_2, delta_y_1), ...]
            approx_pdf_vec_track_el = np.zeros(N) 
            approx_pdf_vec_track_el[0] = sum([P_delta_y_giv_st_vec[0,st]*P_st[st]
                                        for st in np.arange(n)])
            
            def P_delta_y_i_giv_all_prev_delta_y(i):
                ret = sum([P_delta_y_giv_st_vec[i][st] 
                        * P_st_i_giv_all_prev_delta_y_vec[i][st]
                        for st in np.arange(n)])
                return ret
            
            
            def P_delta_y_i_and_st_ip1_giv_st_i(i, st_ip1, st_i): # order-one approx
                if st_ip1 == st_i:
                    return P_dy_giv_sw_0_st(delta_y_track_el[i], st_i)*P_sw_0_giv_st[st_i] 
                return P_dy_giv_sw_1_st1_st2(delta_y_track_el[i], st_i, st_ip1)*P_sw_ge1_giv_st[st_i]*P[st_i,st_ip1]
            
            
            def P_st_i_giv_all_prev_delta_y(st, i):
                ret = sum([P_delta_y_i_and_st_ip1_giv_st_i(i-1, st, st_im1)  
                        * P_st_i_giv_all_prev_delta_y_vec[i-1][st_im1]
                        / approx_pdf_vec_track_el[i-1]
                        for st_im1 in np.arange(n)])
                return ret

            #Now I have defined the functions, I need to use them
            
            for i in range(1, N):
                P_st_i_giv_all_prev_delta_y_vec[i] = np.array([P_st_i_giv_all_prev_delta_y(st, i)
                                                            for st in np.arange(n)])
                approx_pdf_vec_track_el[i] = P_delta_y_i_giv_all_prev_delta_y(i)
            #CHECK INDECES!
            approx_pdf_vec_track.extend(list(approx_pdf_vec_track_el))
        
        return np.array(approx_pdf_vec_track) #[P(delta_y_1), P(delta_y_2 | delta_y_1), P(delta_y_3 | delta_y_2, delta_y_1), ...]+[...]+...

    else:
    
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
    
def n_state_model_pMCMC(delta_t, T, V, Lambda, P, sigma, x0=None, t0=0.0, state='?', seed=1223, checks=False):
    #only varies by sampling an initial x0 and not y0
    """
    delta_t = time step (s)
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
    #print(V, Lambda, P, sigma)

    n = Lambda.shape[0]
    
    if x0 == None:
        #sample x0
        x0 = np.random.normal(loc=0, scale=sigma,
                              size=(1))[0]
        
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
    N_plus_1 = t_points.shape[0]
    # generate error points
    
    x_points = np.zeros(N_plus_1)
    y_points = np.zeros(N_plus_1)
    
    y_points[1:] = np.random.normal(loc=0, scale=sigma, size=N_plus_1-1)
    
    switches = np.zeros(N_plus_1)
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
    Nswitches = switches[1:]
    states = all_states_save[1:]

    return x_points, y_points, t, Nswitches, states


def n_state_model_pMCMC_quick(delta_t, V, Lambda, P, P_st, sigma, x0=None, state='?', seed=1223):
    #only varies by sampling an initial x0 and not y0
    """
    delta_t = time step (s)
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
    #print(V, Lambda, P, sigma)

    n = Lambda.shape[0]
    
    if x0 == None:
        #sample x0
        x0 = np.random.normal(loc=0, scale=sigma,
                              size=(1))[0]
    x = 1.0*x0
    
    #Choosing an initial state
    if state=='?': #state -1 is unknown
        state = random.choices(np.arange(n), weights=P_st, k=1)[0]
    
    t = 0
    
    while t<delta_t: # T is in seconds again
        curr_dt = np.random.exponential(scale=1/Lambda[state])
        if t+curr_dt<delta_t:
            t += curr_dt
            x += curr_dt*V[state]
            state = random.choices(np.arange(n), weights=P[state, :], k=1)[0]
        else:
            x += (delta_t-t)*V[state]
            break
            
    return x, state


def get_data_Ys(delta_t, T, theta, get_parameters, seed, x0=None, t0=0.0, state='?'):
    
    V, Lambda, P, sigma = get_parameters(theta)
    
    # simulate data - only save y
    
    res = n_state_model_pMCMC(delta_t, T, V, Lambda, P, sigma, 
                              x0=x0, t0=t0, state=state, seed=seed)
    #print(res)    
    #only save y
    y_sim = res[1]
        
    return y_sim[1:] #return only y_1,...,y_n, removing y_0=0 (no need to return the rest)

def log_g(Yt, Xt, sigma):
    """log_g - gives back the log probability density of the observation noise model 
               i.e., Y_t ∼ g(Y_t | X_t)
               this corresponds to the weights 
       Y_t - Vector of length N
       X_t - Vector of length N"""
    
    return (-np.power(Yt-Xt, 2)/(2*np.power(sigma, 2))
            + math.log(1/(sigma*math.sqrt(2*math.pi))))

def BootstrapParticleFilter(log_g, f, N_part, delta_t, Yt, sigma, seed):
    """
    BootstrapParticleFilter(log_g,f,X0::Array{Float64,2},Yt::Array{Float64,2},t)
    Generates weighted samples from the distributions π(X_t | Y_1:t-1) for t = 2,...,T
    and produces unbiased estimates of the marginal likelihoods π(Y_t | Y_1:t-1) and
    the full joint likelihood π(Y_1:T).
    Inputs:\n
        `log_g` - log probability density of the observation noise model 
                  i.e., Y_t ∼ g(Y_t | X_t)
        `f` - Simulation process for the state-space model. Given N_part samples {X_t^i}_{i=1}^N_part,
              generates X_{t+1}^i ∼ f(X_{t+1} | X_t^i) for i = 1,2,...,N_part
        `Yt` - time series of observations
        `t` - observation times (assumed exact)
    Outputs:\n
        `log_like` - full joint log likelihood log π(Y_1:T)
        `log_like_t` - T×1 array of marginal log likelihoods log π(Y_t | Y_1:t-1)
        `Xt` - N×N_part×T array of samples with X[:,:,j] is the collection of samples from 
                π(X_j | Y_1:j)
        `log_wt` - N_part×T array of unnormalised log weights, exp(log_wt[i,j]) is the 
                    unnormalised weight of sample Xt[:,i,j]
    """
    #ACTUALLY f(X_{t+1} | X_t^i) is sampled using the state instead!!! 
    
    # determine the number of timesteps
    N_T = len(Yt)

    # initialise memory for particles, weights and marginal log likelihoods
    Xt = np.zeros((N_part, N_T)) # or more generally, Xt = np.zeros((N,N_part,T))
    log_wt = np.zeros((N_part, N_T))
    wtn = np.zeros(N_part)
    wt = np.zeros(N_part)
    log_like_t = -math.inf*np.ones(N_T)
    
    log_like = 0
    
    states = ['?' for i in range(N_part)]
    
    # start particle filter
    for j in range(N_T):
        # propagate particles forward X_t ∼ f(X_t | X_{t-1})
        #delta_t = t[j]-t[j-1] # in our case always 0.3 s
        #print(f(Xt[:,j-1], delta_t, seed+j*7907))
        #print(states)
        if j==0: 
            X0 = np.array([None for _ in range(N_part)])
            Xt[:,0], fin_states = f(delta_t, delta_t, states, X0, seed)
        else:
            Xt[:,j], fin_states = f(delta_t, delta_t, states, Xt[:,j-1], seed+j*7907)
        
        # simplified from Xt[:,j] = f(t[j],Xt[:,j-1],t[j-1]) 
        # simplified from Xt[:,:,j] = f(t[j],Xt[:,:,j-1],t[j-1]) -> delta_t is constant 0.3

        # print('\n\ntime', j, '\nXt', Xt[:,j], '\nYt', Yt[j])
        # set unnormalised log weights
        for i in range(N_part):
            #print(i, Yt[j], Xt[i,j], log_g(Yt[j],Xt[i,j]))
            log_wt[i,j] = log_g(Yt[j],Xt[i,j], sigma) #or log_wt[i,j] = log_g(Yt[:,j],Xt[:,i,j])
            #print(Yt[j],Xt[i,j])
        # compute marginal log-likelihood
        # print(log_wt)
        log_wt_max = np.amax(log_wt) #no axis - this always 0 except last step
        #print('log_wt_max', log_wt_max)
        wt = np.exp(log_wt[:,j] - log_wt_max)
        
        #print('wt', wt)
        #print('np.sum(wt)', np.sum(wt))
        if np.sum(wt) == 0:
            log_like = -math.inf
            break
            
        #print('math.log(np.sum(wt))', math.log(np.sum(wt)))
        log_like_t[j] = math.log(np.sum(wt)) + log_wt_max - math.log(N_part)

        if math.isinf(log_like_t[j]): # break on vanishing marginal likelihood
            log_like = log_like_t[j]
            break
            
        # normalise weights and perform importance resampling
        # to obtain (Xt,wt) ≈ π(Xt | Y_1:t)
        wtn = np.exp(log_wt[:,j] - log_like_t[j])/N_part
        I = random.choices(np.arange(0,N_part), weights=wtn, k=N_part) 
        #print('particle chosen indices', I)
        Xt[0:N_part,j] = Xt[I,j] #or Xt[:,0:M,j] = Xt[:,I,j]   
        states = copy.deepcopy([fin_states[i] for i in I])
        
    # compute joint log likelihood
    if not math.isinf(log_like):
        log_like = np.sum(log_like_t)
    
    return log_like, log_like_t, Xt, log_wt



def log_pi_hat(theta, get_parameters, N_part, delta_t, Yt, seed=0):
    """theta is a vector (m,n) where m=5 in our case since 5 parameters"""
    m = np.array(theta).shape[0] #check
    #print(m) # initial points - one for each particle
    #or some uniform maybe around 0 -> I guess it should not change significantly
    V, Lambda, P, sigma = get_parameters(theta)
    
    n = Lambda.shape[0]
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
    
    def f(delta_t, delta_t_T, states, X_prevs, seed, theta=theta):
        #as we saved the state f(Xt) only depends on the state 
        Xt = np.zeros(N_part)
        new_states = [0 for i in range(N_part)]
        
        for i in range(N_part):
            res = n_state_model_pMCMC_quick(delta_t, V, Lambda, P, P_st, sigma,
                                            x0=X_prevs[i], state=states[i],
                                            seed=seed+1999+i*1997)
            #print(res)
            Xt[i] = res[0]
            #print('Xt for particle', i, 'is', Xt[i])
            new_states[i] = res[1] #final states get updated
            #print('new_state_i', new_states[i])
        #print('All Xt', Xt)
        #print('new_states', new_states)
        return Xt, new_states
        
    log_like, log_like_t, Xt, log_wt = BootstrapParticleFilter(log_g, f, N_part, delta_t, Yt, sigma, seed)

    #def p(theta):
    #    """prior p(theta)"""
    
    # We could use as a prior a uniformly distributed choice of the parameters
    # choose v_F, v_B in [0,4000] and p_FB, p_BF in [0,1]
    # then maybe log_q is not needed since it gets simplified
    # so log_pi_hat <- log_like
    
    return log_like # + log_q which though in my opinion can be simplified




