import numpy as np
from scipy.stats import uniform

class Demon:
    def __init__(self, n , dims = 1,cond="cold",m=1,kb=1):
        self.n = int(n)
        self.init_cond = ["hot","cold"]

        if cond not in self.init_cond:
            raise ValueError('x should not be less than 10!')
        else:
            self.cond=cond

        self.dims = int(dims)
        self.m = m
        self.kb = kb
        self.init_field()
        

        
    def init_field(self,):

        self.demon = 0
        self.sweeps = 0
        self.accepted = 0
        if(self.cond=="cold"):
            self.v=np.ones((self.n,self.dims),dtype=np.float64)*0.5

        self.energy_v=[self.energy(),]
        self.config = [self.v,]
            

    def energy(self,):
        return (0.5*self.m*(self.v*self.v)/self.n).sum()
        
        
    def __call__(self, ):
        '''Print detailed report'''
        print("Average kinetic energy %.4lf\n"%(self.energy()))
        print("Total sweeps %d\n"%(self.sweeps,))
        print("Acceptance rate %.4f\n"%(self.accepted/(self.sweeps*self.n),))

  
          
    def update(self,sweeps=1,delta=1e-2,dump_step=10):

        sw = int(sweeps)
        
        new_v = (2.*uniform.rvs(size=(sw,self.n,self.dims))-1.)*delta

        for s in range(sw):

            for n in range(self.n):

                trial_v = self.v[n]+new_v[s,n]
                
                dE = 0.5*self.m*(trial_v**2-self.v[n]**2).sum()
                
                if( dE<=self.demon ):

                    self.v[n]      = np.copy(trial_v)
                    self.accepted += 1
                    self.demon    -= dE

            if(s%dump_step==0):
                self.config.append(self.v)
            self.energy_v.append(self.energy())
                    
        self.sweeps += sweeps
