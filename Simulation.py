import matplotlib.pyplot as plt 
from scipy.stats import norm
import scipy.stats
import pandas as pd
import numpy as np
i=complex(0,1)

#Get last row of struct
def unpack(st):
    dct = dict()
    for x,y in zip(st[:][-1],st.dtype.names):
        dct[y]=x.tolist()
    return pd.DataFrame(dct)

#Integral approximation via trapezoid method
def trapz(f,a,b,n,*args):
    x = np.linspace(a,b,n)
    y = np.nan_to_num(f(x,*args))
    h = (b-a)/(n-1)
    return (h/2)*(y[1:]+y[:-1]).sum()

#Draw random samples from a multivariate normal distribution
def dWdW(dt,rho):
    mu = np.array([0,0])
    cov = np.array([[1,rho],[rho,1]])
    return np.random.multivariate_normal(mu,dt*cov)

#Draw random samples from Poisson distribution
def dN(dt,lam):
    return np.random.poisson(lam*dt)

class BlackScholes:
    def __init__(self,S_t,sig,tau,r):
        self.S_t = S_t
        self.sig = sig
        self.tau = tau
        self.r = r

    def d1(self,K):
        return (np.log(self.S_t/K)+(self.r+(self.sig**2)/2)*self.tau)/ \
               (self.sig*np.sqrt(self.tau))
    
    def d2(self,K):
        return (np.log(self.S_t/K)+(self.r-(self.sig**2)/2)*self.tau)/ \
               (self.sig*np.sqrt(self.tau))

    def C(self,K):
        return (self.S_t*norm.cdf(self.d1(K)))- \
               (K*np.exp(-self.r*self.tau)*norm.cdf(self.d2(K))) 
                
    def P(self,K):
        return (K*np.exp(-self.r*self.tau)*norm.cdf(-self.d2(K)))- \
               (self.S_t*norm.cdf(-self.d1(K)))
    
    def vega(self,K):
        return (self.S_t*norm.pdf(self.d1(K))*np.sqrt(self.tau))
    
    def delta_C(self,K):
        return norm.cdf(self.d1(K))
    
    def delta_P(self,K):
        return norm.cdf(-self.d1(K))
        
class Heston:
    def __init__(self,kap,the,eta,rho,measure="Q"):
        self.kap = kap
        self.the = the
        self.eta = eta
        self.rho = rho
        self.measure = measure

    #Set state variables
    def set(self,S_t,Y_t,tau,r):
        self.S_t = S_t
        self.Y_t = Y_t
        self.tau = tau
        self.r = r
        self.F = S_t*np.exp(r*tau) 
        return self

    #Characteristic function
    def CF(self,u,j):
        m = ((1+i*u)*self.r*self.tau)/self.the if self.measure == "P" else 0
        b_j = self.kap-j*self.rho*self.eta
        alp_j = -.5*u**2+i*u*(j-.5)
        bet_j = b_j-self.rho*self.eta*i*u
        gam = .5*self.eta**2
        d_j = np.sqrt(bet_j**2-4*alp_j*gam)
        rm_j = (bet_j-d_j)/(2*gam)
        rp_j = (bet_j+d_j)/(2*gam)
        g_j = rm_j/rp_j
        D_j = rm_j*((1-np.exp(-d_j*self.tau))/(1-g_j*np.exp(-d_j*self.tau)))
        C_j = m+self.kap*(rm_j*self.tau-2/self.eta**2* \
                          np.log((1-g_j*np.exp(-d_j*self.tau))/(1-g_j)))
        return np.exp(C_j*self.the+D_j*self.Y_t+i*u*np.log(self.S_t))
    
    #Probability density function
    def PDF(self,K,j):
        def _I(u,k,j): #Integrand
            return np.real(self.CF(u,j)*np.exp(-i*u*k))
        k = np.log(K)
        I = trapz(_I,1e-8,2e2,1000,k,j)
        return 1/np.pi * I
    
    #Cumulative distribution function
    def CDF(self,K,j):
        def _I(u,k,j): #Integrand
            return np.real((self.CF(u,j)*np.exp(-i*u*k))/(i*u))
        k = np.log(K)
        I = trapz(_I,1e-8,2e2,1000,k,j)
        return .5 + 1/np.pi * I      
    
    #Call px
    def C(self,K):
        P0 = self.CDF(K,0) #Probability under risk neutral/real world measure
        P1 = self.CDF(K,1) #Probability under numeraire induced measure
        return self.S_t*P1 - K*np.exp(-self.r*self.tau)*P0
    
    #Put px via put-call parity
    def P(self,K):
        return self.C(K) + K*np.exp(-self.r*self.tau) - self.S_t
    
    #Call delta
    def delta_C(self,K):
        def _I(u,k,j,S): #Integrand
            return np.real((self.CF(u,j)*np.exp(-i*u*k))/S)
        P_1 = self.CDF(K,1)
        k = np.log(K)
        I_1 = trapz(_I,1e-8,2e2,1000,k,1,self.S_t)
        I_0 = trapz(_I,1e-8,2e2,1000,k,0,self.S_t)
        return P_1 + self.S_t/np.pi * I_1 - K/np.pi * I_0
    
    #Put delta via put-call parity
    def delta_P(self,K):
        return self.delta_C(K)-1

class StochasticControl:
    def __init__(self,A,B,gam,mu,r):
        self.A = A
        self.B = B
        self.gam = gam
        self.mu = mu
        self.r = r
        
    #Set state variables
    def set(self,S_t,Y_t,CQ_t,CP_t,del_t,tau):
        self.S_t = S_t
        self.Y_t = Y_t
        self.CQ_t = CQ_t
        self.CP_t = CP_t
        self.del_t = del_t
        self.tau = tau
        return self
        
    #Optimal controls risk neutral market maker
    def RN(self):
        M_0 = self.CQ_t - self.CP_t + self.mu*self.tau*self.S_t*self.del_t
        _x = 2*self.gam-1
        _d = np.sqrt(self.gam**2 * M_0**2 + _x*self.B)
        d_a_L = (_d - self.gam*M_0) / _x
        d_b_L = (_d + self.gam*M_0) / _x
        return d_a_L,d_b_L
        
    #Optimal controls risk averse market maker
    def RA(self,Q1_t,eps):
        M_0 = self.CQ_t - self.CP_t + self.mu*self.tau*self.S_t*self.del_t
        _x = 2*self.gam-1
        _d = np.sqrt(self.gam**2 * M_0**2 + _x*self.B)
        d_a_L = (_d - self.gam*M_0) / _x
        d_b_L = (_d + self.gam*M_0) / _x

        lam_a_L = self.lam(d_a_L)
        lam_b_L = self.lam(d_b_L)
            
        the_4 = -self.tau*self.del_t**2*self.Y_t*self.S_t**2
        the_3 = 2*(lam_a_L-lam_b_L)*self.tau**2*self.del_t**2*self.Y_t*self.S_t**2
            
        M_1 = -the_3+(1-2*Q1_t)*the_4
        M_2 = -the_3-(1+2*Q1_t)*the_4
            
        M_a = M_0 + eps*M_1
        M_b = M_0 + eps*M_2

        d_a_e = (np.sqrt(self.gam**2 * M_a**2 + _x*self.B) - self.gam*M_a) / _x
        d_b_e = (np.sqrt(self.gam**2 * M_b**2 + _x*self.B) + self.gam*M_b) / _x
        return d_a_e,d_b_e

    #Order arrival intensity under square root market impact
    def lam(self,d):
        return self.A/(self.B+d**2)**self.gam
    
class Simulation:
    def __init__(self,params:dict):
        #Other params   
        self.T = params["T"]
        self.dt = params["dt"]
        self.K = params["K"]
        self.M = params["M"] #Number of sims
        self.N = int(self.T/self.dt) 
        self.eps = params["eps"]
        
        #Heston params - measure P
        self.kap_R = params["kap_R"]
        self.the_R = params["the_R"]
        self.eta_R = params["eta_R"]  
        self.rho_R = params["rho_R"]
        
        #Heston params - measure Q
        self.kap_I = params["kap_I"]
        self.the_I = params["the_I"]
        self.eta_I = params["eta_I"]  
        self.rho_I = params["rho_I"]
        
        #Stochastic control params
        self.r = params["r"]
        self.mu = params["mu"]
        self.A = params["A"]
        self.B = params["B"]
        self.gam = params["gam"]
        
        #State process initial values
        self.S_0 = params["S_0"]
        self.Y_0 = params["Y_0"]
        self.X_0 = params["X_0"]
        self.Q1_0 = params["Q1_0"]
        self.Q2_0 = params["Q2_0"]
        
        self.SC = StochasticControl(self.A,self.B,self.gam,self.mu,self.r)
        self.HQ = Heston(self.kap_I,self.the_I,self.eta_I,self.rho_I,"Q")
        self.HP = Heston(self.kap_R,self.the_R,self.eta_R,self.rho_R,"P")
        
        assert 0 < self.eps < 1, "Small parameter is has to be between 0 and 1"
        assert (self.A > 0 and self.B > 0 and self.gam > 1), \
                "A,B has to be larger then 0 and gamma larger then 1"
        assert (2*self.kap_I*self.the_I > self.eta_I**2 and 
                2*self.kap_R*self.the_R > self.eta_R**2), \
                "2*kappa*theta has to be larger then eta^2"
                
    def run(self):
        sp_head = ([(x,np.float64,(self.M,)) for x in ["t","S_t","Y_t","CH_t","CBS_t"]])
        mm_head = ([(x,np.float64,(self.M,)) for x in ["t","X_t","Q1_t","Q2_t","d_a","d_b"]])
        sp = np.zeros(self.N,dtype=sp_head) #State process struct
        rn = np.zeros(self.N,dtype=mm_head) #Risk neutral market maker struct
        ra = np.zeros(self.N,dtype=mm_head) #Risk averse market maker struct
        zi = np.zeros(self.N,dtype=mm_head) #Zero intelligence market maker struct
    
        for m in range(self.M):
            print(f"Path:{m}")
            S_t,Y_t = self.S_0,self.Y_0
            X_RN_t,X_RA_t,X_ZI_t = self.X_0,self.X_0,self.X_0
            Q1_RN_t,Q1_RA_t,Q1_ZI_t = self.Q1_0,self.Q1_0,self.Q1_0
            Q2_RN_t,Q2_RA_t,Q2_ZI_t = self.Q2_0,self.Q2_0,self.Q2_0
            for n in range(self.N):
                t = n*self.dt
                tau = self.T-t
                
                #State process
                dW = dWdW(self.dt,self.rho_I)
                dW_1,dW_2 = dW[1],dW[0]   
                dS = S_t*self.mu*self.dt + S_t*np.sqrt(Y_t)*dW_2 
                dY = self.kap_I*(self.the_I-Y_t)*self.dt + self.eta_I*np.sqrt(Y_t)*dW_1
                Y_t += dY
                S_t += dS
                Y_t=max(Y_t,1e-308) #Bound by float point limit
                
                #Option prices
                self.HQ.set(S_t,Y_t,tau,self.r)
                self.HP.set(S_t,Y_t,tau,self.mu)
                BS = BlackScholes(S_t,np.sqrt(Y_t),tau,self.r)
                CHQ_t = self.HQ.C(self.K) 
                CHP_t = self.HP.C(self.K) 
                CBS_t = BS.C(self.K) 
                self.SC.set(S_t,Y_t,CHQ_t,CHP_t,self.HQ.delta_C(self.K),tau)
                
                #Risk-neutral market maker
                d_RN_a,d_RN_b = self.SC.RN()
                lam_RN_a = self.SC.lam(d_RN_a)
                lam_RN_b = self.SC.lam(d_RN_b)
                dN_RN_a,dN_RN_b = dN(self.dt,lam_RN_a),dN(self.dt,lam_RN_b)
                Q1_RN_t += dN_RN_b - dN_RN_a
                Q2_RN_t = int(-Q1_RN_t*self.HQ.delta_C(self.K)) #Delta hedging
                X_RN_t += (CHQ_t+d_RN_a)*dN_RN_a-(CHQ_t-d_RN_b)*dN_RN_b+Q2_RN_t*dS
                
                #Risk-averse market maker
                d_RA_a,d_RA_b = self.SC.RA(Q1_RA_t,self.eps)
                lam_RA_a = self.SC.lam(d_RA_a)
                lam_RA_b = self.SC.lam(d_RA_b)
                dN_RA_a,dN_RA_b = dN(self.dt,lam_RA_a),dN(self.dt,lam_RA_b)
                Q1_RA_t += dN_RA_b - dN_RA_a
                Q2_RA_t = int(-Q1_RA_t*self.HQ.delta_C(self.K)) #Delta hedging
                X_RA_t += (CHQ_t+d_RA_a)*dN_RA_a-(CHQ_t-d_RA_b)*dN_RA_b+Q2_RA_t*dS
                      
                #Zero-intelligence market maker
                d_ZI = .005 * BS.vega(self.K)
                lam_ZI = self.SC.lam(d_ZI)
                dN_ZI_a,dN_ZI_b = dN(self.dt,lam_ZI),dN(self.dt,lam_ZI)
                Q1_ZI_t += dN_ZI_b - dN_ZI_a
                Q2_ZI_t = -int(Q1_ZI_t*BS.delta_C(self.K)) #Delta hedging
                X_ZI_t += (CHQ_t+d_ZI)*dN_ZI_a - (CHQ_t-d_ZI)*dN_ZI_b + Q2_ZI_t*dS
            
                #Data  
                sp["t"][n,m]=rn["t"][n,m]=ra["t"][n,m]=zi["t"][n,m]=t
                sp["S_t"][n,m],sp["Y_t"][n,m]=S_t,Y_t
                sp["CH_t"][n,m],sp["CBS_t"][n,m]=CHQ_t,CBS_t
                
                rn["X_t"][n,m],rn["d_a"][n,m],rn["d_b"][n,m]=X_RN_t,d_RN_a,d_RN_b
                rn["Q1_t"][n,m],rn["Q2_t"][n,m]=Q1_RN_t,Q2_RN_t
                
                ra["X_t"][n,m],ra["d_a"][n,m],ra["d_b"][n,m]=X_RA_t,d_RA_a,d_RA_b
                ra["Q1_t"][n,m],ra["Q2_t"][n,m]=Q1_RA_t,Q2_RA_t
                
                zi["X_t"][n,m],zi["d_a"][n,m],zi["d_b"][n,m]=X_ZI_t,d_ZI,d_ZI
                zi["Q1_t"][n,m],zi["Q2_t"][n,m]=Q1_ZI_t,Q2_ZI_t
                
        self.sp = unpack(sp)
        self.rn = unpack(rn)
        self.ra = unpack(ra)
        self.zi = unpack(zi)
        return self
    
    #Histograms for terminal values of selected variable across 3 market makers
    def figures(self,var):
        assert self.M > 5, "Not enough paths simulated to get figures" 
        
        #Plot Histograms
        x = self.zi[var]+self.rn[var]+self.ra[var]
        xmx,xmn = max(x),min(x)
        plt.hist(self.zi[var],density=True,bins=30,label="Zero Intelligence",
                 range=(xmn,xmx),alpha=0.6,edgecolor='black')
        plt.hist(self.rn[var],density=True,bins=30,label="Risk Neutral",
                 range=(xmn,xmx),alpha=0.6,edgecolor='black',color="gray")
        plt.hist(self.ra[var],density=True,bins=30,label="Risk Averse",
                 range=(xmn,xmx),alpha=0.6,edgecolor='black')
        mn,mx = plt.xlim()
        plt.xlim(mn,mx)
        plt.legend()
        plt.title(var)
        plt.show()
        
        #Plot density estimators
        x = np.linspace(mn,mx,300)
        kde = scipy.stats.gaussian_kde(self.zi[var])
        plt.plot(x,kde.pdf(x),label="Zero Intelligence")
        kde = scipy.stats.gaussian_kde(self.rn[var])
        plt.plot(x,kde.pdf(x),label="Risk Neutral",color="gray")
        kde = scipy.stats.gaussian_kde(self.ra[var])
        plt.plot(x,kde.pdf(x),label="Risk Averse")
        plt.legend()
        plt.title(var)
        plt.show()

    #Descriptive statistics for terminal values of selected variable across 3 market makers
    def stats(self,var):
        assert self.M > 5, "Not enough paths simulated to get stats" 
        print(f"\n{var}:")
        
        df = pd.DataFrame(columns=["Market maker","Mean","Variance","Skewness","Kurtosis"])
        mm = ["Zero Intelligence","Risk Neutral","Risk Averse"]
        for x,n in zip([self.zi,self.rn,self.ra],range(3)):
            df.loc[n] = [mm[n],x[var].mean(),x[var].var(),x[var].skew(),x[var].kurtosis()]
        print(df)

#-------------------------------------------------------------------------------
params = { 
          "T" : 1/12           #Expiry date
         ,"dt" : (1/12)/500    #Time increment
         ,"K" : 100            #Strike price
         ,"M" : 500            #Number of paths to simulate
         ,"eps" : 0.5          #Small parameter value
    
         ,"kap_R" : 4          #Kappa in Heston model under measure P
         ,"the_R" : 0.04       #Theta in Heston model under measure P
         ,"eta_R" : 0.5        #Eta in Heston model under measure P
         ,"rho_R" : -0.4       #Rho in Heston model under measure P
        
         ,"kap_I" : 4          #Kappa in Heston model under measure Q
         ,"the_I" : 0.04       #Theta in Heston model under measure Q
         ,"eta_I" : 0.5        #Eta in Heston model under measure Q
         ,"rho_I" : -0.4       #Rho in Heston model under measure Q
        
         ,"r" : 0.001          #Spot price process drift under measure Q
         ,"mu" : 0.001         #Spot price process drift under measure P
         ,"A" : 500            #Order arrival intensity parameter A
         ,"B" : 1              #Order arrival intensity parameter B
         ,"gam" : 1.5          #Order arrival intensity parameter gamma
        
         ,"S_0" : 100          #Spot price process initial value
         ,"Y_0" : 0.04         #Variance process initial value
         ,"X_0" : 0            #Cash process initial value
         ,"Q1_0" : 0           #Options inventory process initial value
         ,"Q2_0" : 0           #Stock inventory process initial value
         }

S = Simulation(params)
S.run()
S.figures("X_t")
S.stats("X_t")
S.figures("Q1_t")
S.stats("Q1_t")


    

    
