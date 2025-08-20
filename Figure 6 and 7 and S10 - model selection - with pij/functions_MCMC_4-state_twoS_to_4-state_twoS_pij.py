def run_function(init_seed):
    # importing all the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import copy
    import random
    import pandas as pd
    import math
    import scipy
    from scipy.stats import norm, expon
    from scipy.linalg import null_space
    import pandas as pd
    import time

    def n_state_model(delta_t, T, V, Lambda, P, sigma, x0=0.0, t0=0.0, state='?', seed=1223, checks=False):
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

        if checks:
        
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
                
            if np.any(np.testing.assert_allclose(np.sum(P, axis=1), np.ones(n), rtol=1e-07)):
                print('Error defining transition matrix P sum')
                return None
            
            if np.any(sigma < 0):
                print('Error defining noise parameter sigma')
                return None
            
        else:
            n = Lambda.shape[0]
            
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
        Nswitches = switches[1:]
        states = all_states_save[1:]
        
        return x_points, y_points, t, Nswitches, states


    def get_data_dy(delta_t, T, theta, get_parameters, seed, x0=0.0, t0=0.0, state='?', correlated=True):
        
        V, Lambda, P, sigma = get_parameters(theta)
        
        # simulate data - only save y
        # then save it as delta_y
        if correlated:
            y_sim = n_state_model(delta_t, T, V, Lambda, P, sigma, 
                                x0=x0, t0=t0, state=state, seed=seed)[1] #only save y
        
            delta_Y = np.array([y_sim[i]-y_sim[i-1] for i in range(1,y_sim.shape[0])])
        else:
            N_delta_Y = int(T/delta_t) 
            delta_Y = np.zeros(N_delta_Y)
            for i in range(N_delta_Y):
                y_sim = n_state_model(delta_t, delta_t, V, Lambda, P, sigma, 
                                    x0=x0, t0=t0, state=state, seed=seed+53*i)[1] #only save y
                #print(y_sim)
                delta_Y[i] = y_sim[1]-y_sim[0]

        #print(delta_Y.shape)
            
        return delta_Y #no need to return the rest


    def approx_pdf_up_to_1_switch(V, Lambda, P, sigma, delta_t, delta_y, eps=1e-14, checks=False):
        if checks:
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
                
            if np.any(np.testing.assert_allclose(np.sum(P, axis=1), np.ones(n), rtol=1e-07)):
                print('Error defining transition matrix P sum')
                return None
            
            if np.any(sigma < 0):
                print('Error defining noise parameter sigma')
                return None
        
        else:
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
        
        #print(P_delta_y_giv_st(delta_y, 0))
        #print(P_st[0])

        approx_pdf = sum([P_delta_y_giv_st(delta_y, st)*P_st[st] for st in np.arange(n)])
        
        return approx_pdf


    def approx_pdf_up_to_2_switch(V, Lambda, P, sigma, delta_t, delta_y, eps=1e-14, checks=False):
        if checks:
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
                
            if np.any(np.testing.assert_allclose(np.sum(P, axis=1), np.ones(n), rtol=1e-07)):
                print('Error defining transition matrix P sum')
                return None
            
            if np.any(sigma < 0):
                print('Error defining noise parameter sigma')
                return None
        
        else:
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


    def approx_pdf_track_up_to_1_switch(V, Lambda, P, sigma, delta_t, delta_y_track, eps=1e-14, checks=False):
        if checks:
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
                
            if np.any(np.testing.assert_allclose(np.sum(P, axis=1), np.ones(n), rtol=1e-07)):
                print('Error defining transition matrix P sum')
                return None
            
            if np.any(sigma < 0):
                print('Error defining noise parameter sigma')
                return None
        
        else:
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


    def approx_pdf_theta(theta, get_parameters, delta_t, delta_y, up_to_switches = 1, track = False, checks=False):
        
        V, Lambda, P, sigma = get_parameters(theta)
        
        #up-to-one switch
        if up_to_switches == 1:
            if track:
                return approx_pdf_track_up_to_1_switch(V, Lambda, P, sigma, delta_t, delta_y, checks=checks)
            else:
                return approx_pdf_up_to_1_switch(V, Lambda, P, sigma, delta_t, delta_y, checks=checks)
            
        #up-to-two switches
        elif up_to_switches == 2:
            if track:
                print('Error')
            else:
                return approx_pdf_up_to_2_switch(V, Lambda, P, sigma, delta_t, delta_y, checks=checks)
            
        else:
            print('Error')
            
        return None


    def log_pi_hat(theta, get_parameters, delta_t, delta_y, up_to_switches = 1, track = False):
        
        log_like = np.sum(np.log(approx_pdf_theta(theta, get_parameters, delta_t, delta_y,
                                                up_to_switches = up_to_switches,
                                                track = track)))
        if np.isinf(log_like) or np.isnan(log_like):
            return -np.inf
        
        return log_like # + log_q which can be simplified


    def MetropolisHastings(q, cov_matrices, theta_0, get_parameters, burnin, n_after_burnin, n_chains,
                        delta_t, delta_Y, seed, up_to_switches = 1, track = False, n_tracks = 1):
        """
        Generates n interations of the a Markov Chain with stationary distribution 
        pi(theta) using the Metropolis-Hastings MCMC algorithm.
        
        Inputs:
            log_pi_hat - natural logarithm of unbiased estimator to the target 
                        propability density function ln pi(theta). 
                        pi(theta) need only be known up to a constant of proportionality.
            q - proposal density sampler, it generates samples
            theta_0 - initial condition for the Markov Chain 
            N - number of samples used to for the estimator log_pi_hat
            burnin - number of iterations to discard as burn-in samples
            n - number of iterations to perform (after burn-in samples)
            #NOT INCLUDING log_q - natural logarithm of proposal density ln q(theta*|theta) - as we define it symmetric
            
        Outputs:
            theta_t - array of samples theta_t[:,i] is the Markov Chain state at the i-th 
                    iteration. theta[j,:] is the trace of the j-th dimension of theta_t.
        """
        np.random.seed(seed=seed)
        
        # Get dimenisionality of state space
        theta_0 = np.array(theta_0)
        n_param = theta_0.shape[0]
        if theta_0.shape[1] != n_chains:
            print('Error', n_chains, 'initial conditions should be given')

        # initialise previous log pi(theta) to avoid repeat evaluation
        prev_log_pi = np.zeros(n_chains)
        for ind in range(n_chains):
            if n_tracks == 1:
                prev_log_pi[ind] = log_pi_hat(theta_0[:,ind], get_parameters, delta_t, delta_Y,
                                            up_to_switches = up_to_switches,
                                            track = track)
            else:
                for delta_y in delta_Y:
                    prev_log_pi[ind] += log_pi_hat(theta_0[:,ind], get_parameters, delta_t, delta_y,
                                                up_to_switches = up_to_switches,
                                                track = track)
        cur_log_pi = 1.0*prev_log_pi

        # allocate memory for theta_t
        theta_t = np.zeros((n_param, n_chains, n_after_burnin))
        log_pi = np.zeros((n_chains, n_after_burnin))
        # initialise theta_t
        theta_t[:,:,0] = 1.0*theta_0
        theta_t[:,:,1] = 1.0*theta_0

        log_pi[:,0] = 1.0*prev_log_pi
        log_pi[:,1] = 1.0*prev_log_pi

        # generate array of u ~ U(0,1) for 
        # sample uniform and broadcast log
        log_u = np.log(np.random.uniform(0, 1, (n_chains, burnin+n_after_burnin))) # sample uniform
        
        j = 1
        # perform Metropolis-Hastings interations
        for i in range(1, burnin+n_after_burnin):
            """if i%1000==0:
                print('iteration', i, 'out of', burnin+n_after_burnin)"""
            # In the burnin the parameters theta are not modified
            if i <= burnin:
                #  burn-in period, set prev to j=0 and cur to j=1
                j = 1
                theta_t[:,:,j-1] = theta_t[:,:,j]
                log_pi[:,j-1] =  1.0*prev_log_pi
                #print(theta_t[:,j-1])
            else: # offset MCMC index
                j = i - burnin
            
            #print(theta_p)
            for ind in range(n_chains):        
                # generate proposal theta_p ~ q(⋅ | theta_j)
                # theta_p is the proposed theta (also defined theta_star)
                #print('j-1', theta_t[:,j-1])
                theta_curr = 1.0*theta_t[:,ind,j-1]
                theta_p = q(theta_curr, cov_matrices[ind,:,:])
                # compute log pi(theta_p)
                if n_tracks == 1:
                    cur_log_pi[ind] = log_pi_hat(theta_p, get_parameters, delta_t, delta_Y,
                                            up_to_switches = up_to_switches,
                                            track = track)
                else:
                    cur_log_pi[ind] = 0.0
                    for delta_y in delta_Y:
                        cur_log_pi[ind] += log_pi_hat(theta_p, get_parameters, delta_t, delta_y,
                                                    up_to_switches = up_to_switches,
                                                    track = track)
                
                """# compute acceptance probability (in log form)
                log_alpha = min(0.0, cur_log_pi + log_q(theta_t[:,j-1],theta_p[:]) 
                                    - prev_log_pi - log_q(theta_p[:],theta_t[:,j-1]))
                note that if q(theta1, theta2) = q(theta2, theta1) (symmetric)
                then it is gets simplified, hence"""
                # compute acceptance probability (in log form)
                # print(cur_log_pi[ind]-prev_log_pi[ind])
                log_alpha = min(0.0, cur_log_pi[ind] - prev_log_pi[ind])
            
                # accept transition with prob alpha
                if log_u[ind,i] <= log_alpha:
                    theta_t[:,ind,j] = theta_p
                    # store for next iteration
                    prev_log_pi[ind] =  1.0*cur_log_pi[ind]
                else: # reject transition with prob 1 - alpha
                    theta_t[:,ind,j] = theta_curr

            log_pi[:,j] = cur_log_pi
                
        return theta_t, log_pi


    def compute_effective_n_of_simulation_draws(n_tot_hat, m, psi, var_hat_plus):
        #Variogram
        V_t = np.zeros(n_tot_hat)
        for t in range(n_tot_hat-1):
            V_t[t] = sum([sum([(psi[j,i] - psi[j, i-t])**2 for i in range(t+1, n_tot_hat)])
                        for j in range(m)])/(m*(n_tot_hat-t))
        
        rho_hat_t = 1 - V_t/(2*var_hat_plus)
        
        # Find minimum T such that rho_hat_t[T+1] + rho_hat_t[T+2] < 0
        T = n_tot_hat-2
        for t in range(0, n_tot_hat-2):
            if rho_hat_t[t+1] + rho_hat_t[t+2] < 0:
                T = t
                break
        
        n_hat_eff = (m*n_tot_hat)/(1 + 2*np.sum(rho_hat_t[:T+1]))

        return n_hat_eff


    def check_convergence(theta, split = False, compute_n_eff = False):
        theta = np.array(theta)
        n_param, n_sim, n_tot = theta.shape
        #print('theta shape in check_convergence', theta.shape)

        if n_sim == 1 or split:
            #split!
            if n_tot%2!=0:
                n_tot -= 1
            m = 2*n_sim
            n_tot_hat = n_tot//2
            mkn_theta = np.zeros((n_param,m,n_tot_hat))
            for j in range(n_sim):
                mkn_theta[:,2*j,:] = np.array([theta[:,j,:n_tot_hat]])
                mkn_theta[:,2*j+1,:] = np.array([theta[:,j,n_tot_hat:n_tot]])
            theta = mkn_theta
        else:
            m = n_sim
            n_tot_hat = n_tot
                
        R_hat = np.zeros(n_param)
        n_hat_eff = np.zeros(n_param)
        
        for k in range(n_param):
            psi = theta[k,:,:]
            psi_bar_j = np.zeros(m)
            s_j_sq = np.zeros(m)
            for j in range(m):
                psi_bar_j[j] = np.mean(psi[j,:])
                s_j_sq[j] = sum([(psi[j,i] - psi_bar_j[j])**2 for i in range(n_tot_hat)]) / (n_tot_hat-1) 
            psi_bar = np.mean(psi_bar_j)
            
            #Between-sequence variance
            B = sum([(psi_bar_j[j] - psi_bar)**2 for j in range(m)]) * n_tot_hat / (m-1)
            
            #Within-sequence variance
            W = np.mean(s_j_sq)
            
            #Estimate marginal posterior variance: var(psi | y)
            #Using an overestimation: 
            #assumes starting distribution is approprietely overdispersed
            #but unbiased under stationarity
            var_hat_plus = (W * (n_tot_hat-1) + B) / n_tot_hat
            
            #R hat: needs to be < 1.1 for each param
            R_hat[k] = math.sqrt(var_hat_plus / W)

            if compute_n_eff:
                n_hat_eff[k] += compute_effective_n_of_simulation_draws(n_tot_hat, m, psi, var_hat_plus)

        if compute_n_eff:
            return R_hat, n_hat_eff
        
        return R_hat, None

    #plotting everything - more helpful to just plot the grid
    def plots_mult(theta, theta_true, j1, n_param, parameter_names_tex):
        #plots
        theta = np.array(theta)

        print('Iteration'+str(j1))
        fig, ax = plt.subplots(n_param, figsize=(n_param*2,n_param*2))
        for i in range(n_param):
            ax[i].hist(theta[i,:], bins=50,
                    label=parameter_names_tex[i]+' distribution', color='lightsteelblue')
            ax[i].axvline(theta_true[i], color='black',
                        label='true value '+parameter_names_tex[i])
            ax[i].legend()
        plt.show()

        for i in range(n_param):
            plt.figure()
            plt.title('Check convergence of '+parameter_names_tex[i])
            plt.plot(np.arange(theta[i,:].shape[0]), theta[i],
                    label='simulation')
            plt.axhline(theta_true[i], color='black', label='true value')
            plt.xlabel('simulation number')
            plt.ylabel(parameter_names_tex[i])
            plt.legend()
            plt.show()

        plt.figure()
        for i in range(1,n_param):
            for j in range(i):
                plt.title('Checking correlations')
                plt.plot(theta[i,:], theta[j,:], '.', alpha=0.1)
                plt.xlabel(parameter_names_tex[i])
                plt.ylabel(parameter_names_tex[j])
                plt.show()

        return None
                

    #saving plot grid with univariate and multivariate posteriors       
    def plots_2_mult(theta, theta_true, j1, n_param, parameter_names_tex, log_pi = None, titlestr='', plot_median = True):
        theta = np.array(theta)
        #plots
        fig, ax = plt.subplots(n_param, n_param, figsize=(n_param*2+5,n_param*2+2))
        plt.subplots_adjust(wspace=0.35, hspace=0.2)
        for i in range(n_param):
            ax[i,i].set_ylabel(parameter_names_tex[i])
            ax[i,i].set_xlabel(parameter_names_tex[i])
        
        for i in range(n_param):
            ax[i,i].hist(theta[i,:], bins=25,
                        label=r'$\theta$ distribution', color='lightsteelblue')
            if theta_true is not None:
                ax[i,i].axvline(theta_true[i], color='black',
                                label=r'true $\theta$')
            if log_pi is not None:
                best_theta_comp_i = theta[i, np.nanargmax(log_pi)]
                ax[i,i].axvline(best_theta_comp_i, linestyle='-.', color='blue',
                                label=r'best $\theta$')
            if plot_median:
                ax[i,i].axvline(np.median(theta[i,:]), linestyle='--', color='red',
                                label=r'median $\theta$')
        for i in range(n_param):
            for j in range(0,i):
                ax[i,j].axis('off')
            for j in range(i+1,n_param):
                #plt.title('Checking correlations')
                ax[i,j].hist2d(theta[j,:], theta[i,:], density=True, bins=30,
                            norm=colors.LogNorm(), cmap='Blues')
        plt.subplots_adjust(right=0.85)
        plt.legend(bbox_to_anchor=(-0.50, 0.7), loc='upper right', borderaxespad=0)
        plt.savefig(titlestr+"parameters_posteriors"+str(j1)+".png", format="png", dpi=1200, bbox_inches="tight")
        plt.show()

        return None


    #plotting everything - more helpful to just plot the grid
    def plots_mult_data(theta, j1, n_param, parameter_names_tex):
        #plots
        theta = np.array(theta)

        print('Iteration'+str(j1))
        fig, ax = plt.subplots(n_param, figsize=(10,10))
        for i in range(n_param):
            ax[i].hist(theta[i,:], bins=50,
                    label=parameter_names_tex[i]+' distribution', color='lightsteelblue')
            ax[i].legend()
        plt.show()

        for i in range(n_param):
            plt.figure()
            plt.title('Check convergence of '+parameter_names_tex[i])
            plt.plot(np.arange(theta[i,:].shape[0]), theta[i],
                    label='simulation')
            plt.xlabel('simulation number')
            plt.ylabel(parameter_names_tex[i])
            plt.legend()
            plt.show()

        plt.figure()
        for i in range(1,n_param):
            for j in range(i):
                plt.title('Checking correlations')
                plt.plot(theta[i,:], theta[j,:], '.', alpha=0.1)
                plt.xlabel(parameter_names_tex[i])
                plt.ylabel(parameter_names_tex[j])
                plt.show()

        return None
        
    #Plot data fits    
    def plot_data_fit(theta_true, get_parameters, delta_t, delta_y, theta_comp, log_pi, i_sim, up_to_switches = 1):
        
        eval_points = np.linspace(np.min(delta_y), np.max(delta_y), 100)

        modes_theta_comp = theta_comp[:, np.nanargmax(log_pi)]
        approx_pdf_comp = approx_pdf_theta(modes_theta_comp, get_parameters, delta_t, eval_points, up_to_switches = up_to_switches, track = False)
        
        plt.figure(figsize=(5,4))
        plt.hist(delta_y, bins=min(int(delta_y.shape[0]/10), 100), density=True, color='orange', label='data '+r'$P(\Delta y | \theta)$')

        if theta_true is not None:
            approx_pdf_true = approx_pdf_theta(theta_true, get_parameters, delta_t, eval_points, up_to_switches = up_to_switches, track = False)
            plt.plot(eval_points, approx_pdf_true, color='black', label=str(r'true $\theta$'))
        plt.plot(eval_points, approx_pdf_comp, color='blue', label=str(r'best $\theta$'))
        plt.legend()
        plt.xlabel(r'$\Delta y$')
        plt.ylabel(r'$P_m(\Delta y | \theta),\ m=$'+str(up_to_switches))
        plt.savefig("fits_compared_to_data_"+str(int(i_sim))+".png", format="png", dpi=1200, bbox_inches="tight")
        
        plt.show()

        return None
        
        
    def plot_data_fit_median(theta_true, get_parameters, delta_t, delta_y, theta_comp, log_pi = None, i_sim = 0, up_to_switches = 1, nbins = None):
        
        eval_points = np.linspace(np.min(delta_y), np.max(delta_y), 100)
        
        approx_pdf_comp_median = approx_pdf_theta(np.median(theta_comp, axis=1), get_parameters, delta_t,
                                                eval_points, up_to_switches = up_to_switches, track = False)
        
        print('theta median', get_parameters(np.median(theta_comp, axis=1)))
        if nbins == None:
            nbins = min(int(delta_y.shape[0]/10), 100)

        plt.figure(figsize=(5,4))
        plt.hist(delta_y, bins=nbins, density=True, color='orange', label='data '+r'$P(\Delta y | \theta)$')
        if theta_true is not None:
            approx_pdf_true = approx_pdf_theta(theta_true, get_parameters, delta_t, eval_points, up_to_switches = up_to_switches, track = False)
            plt.plot(eval_points, approx_pdf_true, color='black', label=str(r'true $\theta$'))
        if log_pi is not None:
            modes_theta_comp = theta_comp[:, np.nanargmax(log_pi)]
            approx_pdf_comp = approx_pdf_theta(modes_theta_comp, get_parameters, delta_t, eval_points, up_to_switches = up_to_switches, track = False)
            print('theta best', get_parameters(modes_theta_comp))
            plt.plot(eval_points, approx_pdf_comp, '-.', color='blue', label=str(r'best $\theta$'))
        plt.plot(eval_points, approx_pdf_comp_median, linestyle='--', color='red', label=str(r'median $\theta$'))
        plt.legend()
        plt.xlabel(r'$\Delta y$')
        plt.ylabel(r'$P_m(\Delta y | \theta),\ m=$'+str(up_to_switches))
        plt.savefig("fits_compared_to_data_median_"+str(i_sim)+".png", format="png", dpi=1200, bbox_inches="tight")
        
        plt.show()


    def plot_data_best_fit(theta_true, get_parameters, delta_t, delta_y, theta_comp, log_pi = None, i_sim = 0, up_to_switches = 1, nbins = None):
        
        eval_points = np.linspace(np.min(delta_y), np.max(delta_y), 100)
        
        if nbins == None:
            nbins = min(int(delta_y.shape[0]/10), 100)

        plt.figure(figsize=(5,4))
        plt.hist(delta_y, bins=nbins, density=True, color='orange', label='data '+r'$P(\Delta y | \theta)$')
        if theta_true is not None:
            approx_pdf_true = approx_pdf_theta(theta_true, get_parameters, delta_t, eval_points, up_to_switches = up_to_switches, track = False)
            plt.plot(eval_points, approx_pdf_true, color='black', label=str(r'true $\theta$'))
        if log_pi is not None:
            modes_theta_comp = theta_comp[:, np.nanargmax(log_pi)]
            approx_pdf_comp = approx_pdf_theta(modes_theta_comp, get_parameters, delta_t, eval_points, up_to_switches = up_to_switches, track = False)
            print('theta best', get_parameters(modes_theta_comp))
            plt.plot(eval_points, approx_pdf_comp, '-.', color='blue', label=str(r'best $\theta$'))
        plt.legend()
        plt.xlabel(r'$\Delta y$')
        plt.ylabel(r'$P_m(\Delta y | \theta),\ m=$'+str(up_to_switches))
        plt.savefig("best_fit_compared_to_data_"+str(i_sim)+".png", format="png", dpi=1200, bbox_inches="tight")
        
        plt.show()


    def run_MCMC(q, cov_matrices, theta_true, theta_init, get_parameters, delta_Y, parameter_names, burnin = 1000,
                n_after_burnin = 10000,
                delta_t = 0.3, n_chains = 4,
                seeds_list = [1229, 1231, 1237, 1249],
                plots = True, up_to_switches = 1, track = False, bool_check_convergence = True, n_tracks = 1):
        
        #print('n_chains =', n_chains)
        #print(delta_Y)

        save_theta, save_log_pi = MetropolisHastings(q, cov_matrices, theta_init, get_parameters,
                                                        burnin = burnin,
                                                        n_after_burnin = n_after_burnin, n_chains = n_chains,
                                                        delta_t = delta_t, delta_Y = delta_Y,
                                                        seed = seeds_list[0],
                                                        up_to_switches = up_to_switches,
                                                        track = track, n_tracks = n_tracks)
        
        #print('save_theta.shape =', save_theta.shape)
        print('done')

        if bool_check_convergence:
            R_hat, n_hat_eff = check_convergence(save_theta)
        else:
            R_hat, n_hat_eff = None, None


        if plots: 
            for ind in range(n_chains):
                #plots
                fig, ax = plt.subplots(5, figsize=(10,10))
                for i in range(5):
                    ax[i].hist(save_theta[ind][i], bins=50,
                                label=parameter_names[i]+' distribution')
                    ax[i].axvline(theta_true[i], color='black',
                                    label='true value '+parameter_names[i])
                    ax[i].legend()
                plt.show()

                for i in range(5):
                    plt.figure()
                    plt.title('Check convergence of '+parameter_names[i])
                    plt.plot(np.arange(save_theta[ind][i].shape[0]), save_theta[ind][i],
                                label='simulation')
                    plt.axhline(theta_true[i], color='black', label='true value')
                    plt.xlabel('simulation number')
                    plt.ylabel(parameter_names[i])
                    plt.legend()
                    plt.show()

                plt.figure()
                plt.title('Checking correlations')
                plt.plot(save_theta[ind][0], save_theta[ind][1], '.', alpha=0.1)
                plt.xlabel(parameter_names[0])
                plt.ylabel(parameter_names[1])
                plt.show()

                plt.figure()
                plt.title('Checking correlations')
                plt.plot(save_theta[ind][2], save_theta[ind][3], '.', alpha=0.1)
                plt.xlabel(parameter_names[2])
                plt.ylabel(parameter_names[3])
                plt.show()

        #print(save_theta)
        
        return save_theta, save_log_pi, R_hat, n_hat_eff



    def distinct_track_runs_MCMC(theta_true, n_param = None, get_parameters = None, parameter_names = None, parameter_names_tex = None,
                                burnin = 1000, n_after_burnin = 10000,
                                delta_t = 0.3, T = 60, n_chains = 4, n_sim = 10,
                                plots = False, save = True, all_plots = False,
                                plot_posteriors_grid = True, plot_fit = True,
                                plot_fit_median = True, track = False,
                                up_to_switches = 1, init_cov_matrix = None, init_cov_matrices = None, q = None,
                                seeds_list = [1229, 1231, 1237, 1249],
                                delta_Y = None, theta_init = None, theta_init_distribution = None,
                                resample_delta_Y = False, correlated = True, n_adaptive_updates = 10, init_seed = 0, n_tracks = 1,
                                show_time = False):
        # to evaluate separate tracks give delta_Y as a list of np.array
        if n_param is None:
            n_param = len(parameter_names)

        all_theta = np.zeros((n_param, n_sim, n_chains, n_after_burnin))
        all_log_pi = np.zeros((n_sim, n_chains, n_after_burnin))
        all_R_hat = np.zeros((n_param, n_sim))
        all_n_hat_eff = []
        i_sim = 0
        n_skipped = 0
        all_delta_Y = []
        already_skipped_theta_init = 0

        if init_cov_matrix is None:
            if init_cov_matrices is None:
                if n_chains == 1:
                    init_cov_matrix = np.diag(0.01*np.ones(theta_true.shape))
                else:
                    init_cov_matrix = np.array([np.diag(0.01*np.ones(theta_true.shape)) for _ in range(n_chains)])
            else:
                init_cov_matrix = np.array([np.diag(init_cov_matrices[:,i_sim]) for _ in range(n_chains)])

        if q is None:
            print('Error: resampling matrix q not defined')
            return None
        
        if delta_Y is None:
            data_seed = 3001+4999*init_seed
            if n_tracks == 1:
                delta_Y = get_data_dy(delta_t = delta_t, T = T, theta = theta_true,
                                    get_parameters = get_parameters, seed = data_seed, correlated = correlated)
            else:
                delta_Y = []
                for i in range(n_tracks):
                    delta_Y.append(np.array(get_data_dy(delta_t = delta_t, T = T, theta = theta_true,
                                                        get_parameters = get_parameters, seed = data_seed+5099*i, correlated = correlated)))
        #print(delta_Y)        
        #print(cov_matrix)

        if theta_init_distribution is not None:
            theta_init_i_sim = np.array([theta_init_distribution(4759+257*(((i_sim+n_skipped+init_seed)*4)+i)) for i in range(n_chains)]).T
        else:
            theta_init_i_sim = np.array([theta_init[:,i_sim] for _ in range(n_chains)]).T

        cov_matrix = 1.0*init_cov_matrix

        start = time.time()

        while i_sim < n_sim:

            print('Iteration', i_sim)
            print('Running the burnin and adaptively computing the covariance matrices for each chain.')
            for index in range(n_adaptive_updates):
                #print("theta init for adaptive step", index, "is", theta_init_i_sim)
                theta0, log_pi0 = MetropolisHastings(q, cov_matrix, theta_init_i_sim,
                                                        get_parameters = get_parameters,
                                                        burnin = 0,
                                                        n_after_burnin = burnin//n_adaptive_updates,
                                                        n_chains = n_chains,
                                                        delta_t = delta_t, delta_Y = delta_Y,
                                                        seed = 3001 + 7901*(i_sim+n_skipped+1+init_seed),
                                                        up_to_switches = up_to_switches,
                                                        track = track, n_tracks = n_tracks)
                #updating covariance matrix
                cov_matrix = np.array([((2.38**2)/n_param)*np.cov(theta0[:,ind,:]) for ind in range(n_chains)])

                #updating theta_init
                theta_init_i_sim = theta0[:,:,-1].reshape((n_param,n_chains))

            #print(theta.shape, theta_init.shape)
            print('Computed the covariance matrices.')
            print('Running MCMC with', n_chains, 'chains')

            theta, log_pi, R_hat, n_hat_eff = run_MCMC(q = q, cov_matrices = cov_matrix, theta_true = theta_true,
                                                        theta_init = theta_init_i_sim,
                                                        get_parameters = get_parameters,
                                                        delta_Y = delta_Y,
                                                        parameter_names = parameter_names,
                                                        burnin = 0,
                                                        n_after_burnin = n_after_burnin,
                                                        delta_t = delta_t,
                                                        n_chains = n_chains,
                                                        seeds_list = seeds_list,
                                                        plots = plots,
                                                        up_to_switches = up_to_switches, 
                                                        track = track, n_tracks = n_tracks)
            
            print('R_hat =', R_hat)
            #print('n_hat_eff =', n_hat_eff)

            #print(theta)
            #print(log_pi)
            
            if np.all(R_hat < 1.1):
                all_theta[:, i_sim, :, :] = 1.0*theta
                all_log_pi[i_sim, :, :] = 1.0*log_pi
                all_R_hat[:, i_sim] = 1.0*R_hat
                all_n_hat_eff.append(n_hat_eff)
                all_delta_Y.append(delta_Y)

                if all_plots:
                    plots_mult(theta.reshape((n_param, n_chains*n_after_burnin)), theta_true, i_sim, n_param, parameter_names_tex)
                
                if plot_posteriors_grid:
                    plots_2_mult(theta.reshape((n_param, n_chains*n_after_burnin)),
                                theta_true, i_sim, n_param, parameter_names_tex, log_pi[0])

                if not track:
                    if plot_fit:
                        plot_data_fit(theta_true, get_parameters, delta_t, delta_Y,
                                    theta.reshape((n_param, n_chains*n_after_burnin)), np.array(log_pi[0]), i_sim,
                                    up_to_switches = up_to_switches)
                    
                    if plot_fit_median:
                        plot_data_fit_median(theta_true, get_parameters, delta_t, delta_Y,
                                            theta.reshape((n_param, n_chains*n_after_burnin)), np.array(log_pi[0]), i_sim,
                                            up_to_switches = up_to_switches)
                
                print("Done simulation", i_sim)

                i_sim += 1
                
                if i_sim < n_sim:
                    if resample_delta_Y:
                        data_seed = 3001 + 4999*(i_sim+init_seed) #or (i_sim+n_skipped)
                        if n_tracks == 1:
                            delta_Y = get_data_dy(delta_t = delta_t, T = T, theta = theta_true,
                                                get_parameters = get_parameters, seed = data_seed, correlated = correlated)
                        else:
                            delta_Y = []
                            for i in range(n_tracks):
                                delta_Y += list(get_data_dy(delta_t = delta_t, T = T, theta = theta_true,
                                                            get_parameters = get_parameters, seed = data_seed+5099*i, correlated = correlated))
                                    
                    if theta_init_distribution is not None:
                        theta_init_i_sim = np.array([theta_init_distribution(4759+257*(((i_sim+n_skipped+init_seed)*4)+i)) for i in range(n_chains)]).T
                    else:
                        theta_init_i_sim = np.array([theta_init[:,i_sim] for _ in range(n_chains)]).T
                    
                    already_skipped_theta_init = 0

                    if init_cov_matrices is not None:
                        init_cov_matrix = np.array([np.diag(init_cov_matrices[:,i_sim]) for _ in range(n_chains)])

                    cov_matrix = 1.0*init_cov_matrix

            else:
                n_skipped +=1
                already_skipped_theta_init += 1
                if already_skipped_theta_init == 3 and theta_init_distribution is not None:
                    print("The chains have failed to converge three times with this sampled initial distribution. Sampling a new one.")
                    theta_init_i_sim = np.array([theta_init_distribution(4759+257*(((i_sim+n_skipped+init_seed)*4)+i)) for i in range(n_chains)]).T
                    already_skipped_theta_init = 0
                else:
                    theta_init_i_sim = theta[:,:,-1].reshape((n_param, n_chains))
                #print("Not converged:", theta, log_pi, R_hat, n_hat_eff)
        end = time.time()
        if show_time:
            print("The total runtime is", end-start, "seconds.")
        print("Not converged:", n_skipped)

        all_delta_Y = np.array(all_delta_Y)
        all_n_hat_eff = np.array(all_n_hat_eff)
        
        #SAVING RESULTS
        all_theta_i = [np.array(all_theta[i,:,:,:].reshape((n_sim, n_chains*n_after_burnin))) for i in range(n_param)]

        if save:
            for i in range(n_param):
                (pd.DataFrame(all_theta_i[i]).to_csv('MCMC, seed '+str(init_seed)+', parameter '
                                                    +parameter_names[i]
                                                    +', burnin='
                                                    +str(burnin)+', n_after_burnin='
                                                    +str(n_after_burnin)))
            (pd.DataFrame(all_log_pi.reshape((n_sim, n_chains*n_after_burnin))).to_csv('MCMC, seed '
                                                                                    +str(init_seed)
                                                                                    +', log_pi '
                                                                                        +', burnin='
                                                                                        +str(burnin)
                                                                                    +', n_after_burnin='
                                                                                    +str(n_after_burnin)))

        return all_theta, all_log_pi, all_R_hat, all_n_hat_eff, all_delta_Y



    # define variables
    data_seed = 1223
    burnin = 1000 #10k
    n_after_burnin = 10000 #10k
    delta_t = 0.3

    #T=60 gives 200 delta_y
    T = 120
    n_chains = 4
    n_sim = 1

    n_adaptive_updates = 5

    # 4-state model with Erlang distribution
    V_F = 2000
    V_B = -1500
    V = np.array([V_F, V_B])
    Lambda = np.array([1, 0.5, 2, 0.15])
    log_Lambda = np.log(Lambda)
    P = np.array([[0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.25, 0.25, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.0]])
    Pij = np.array([0.25, 0.25])
    sigma = 50.0
    n_param = 9


    parameter_names = ['v1', 'v2', 'log(lambda1)', 'log(lambda2)',
                    'log(lambda3)', 'log(lambda4)', 'p31', 'p32', 'sigma']
    parameter_names_tex = [r'$v_1$', r'$v_2$', r'log$(\lambda_1)$',
                        r'log$(\lambda_2)$', r'log$(\lambda_3)$',
                        r'log$(\lambda_4)$', 
                        r'$p_{31}$', r'$p_{32}$',
                        r'$\sigma$']

    #choose initial covariance matrix for resampling
    init_cov_matrix = np.array([np.diag(np.array([1000, 1000, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10]))
                                for _ in range(n_chains)])


    correlated = True
    up_to_switches = 1
    track = True

    plots = False
    save = True
    all_plots = False
    plot_posteriors_grid = False
    plot_fit = False
    plot_fit_median = False

    theta_true = list(V) + list(log_Lambda) + list(Pij) + [sigma] #not including values for P for 2x2

    #4-STATE MODEL
    def get_parameters(theta):
        """Obtaining parameters from theta"""
        #k=2
        V = np.array(list(theta[0:2])+[0.0, 0.0]) 
        Lambda = np.exp(list(theta[2:6]))
        P = np.array([[0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [theta[6], theta[7], 0.0, 1-theta[6]-theta[7]],
                    [0.0, 0.0, 1.0, 0.0]])
        sigma = 1.0*theta[-1]
        #print(V, Lambda, P, sigma)
        return V, Lambda, P, sigma

    data_seed = 3001+4999*init_seed
    i = 0
    delta_Y = get_data_dy(delta_t = delta_t, T = T, theta = theta_true,
                        get_parameters = get_parameters, seed = data_seed+5099*i, correlated = correlated)

    def q(theta, cov_matrix=init_cov_matrix):
        """q samples a new theta_star given theta"""
        #print(theta.shape)
        theta_star = np.random.multivariate_normal(theta, cov_matrix)
        while (theta_star[0]<0 or theta_star[0]>2.0*V_F or 
            theta_star[1]>0 or theta_star[1]<2.0*V_B or 
            np.any(theta_star[2:6]<-4) or np.any(theta_star[2:6]>4) or
            np.any(theta_star[6:8])<0 or (theta_star[6]+theta_star[7])>1 or 
            theta_star[-1]<0 or theta_star[-1]>2.0*sigma):
            theta_star = np.random.multivariate_normal(theta, cov_matrix)
        return theta_star


    def theta_TRUE(seed):
        return theta_true

    distinct_track_runs_MCMC(theta_true = theta_true,
                                    get_parameters = get_parameters,
                                    parameter_names = parameter_names,
                                    parameter_names_tex = parameter_names_tex,
                                    burnin = burnin, n_after_burnin = n_after_burnin,
                                    delta_t = delta_t, T = T,
                                    n_chains = n_chains, n_sim = n_sim,
                                    plots = plots, save = save, all_plots = all_plots,
                                    plot_posteriors_grid = plot_posteriors_grid,
                                    plot_fit = plot_fit,
                                    plot_fit_median = plot_fit_median, track = track,
                                    up_to_switches = up_to_switches,
                                    init_cov_matrix = init_cov_matrix, q = q,
                                    delta_Y = delta_Y, init_seed = init_seed,
                                    theta_init = None, n_adaptive_updates = n_adaptive_updates,
                                    theta_init_distribution = theta_TRUE,
                                    correlated = correlated, show_time = True)