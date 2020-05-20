import numpy as np
from scipy.stats import uniform
from itertools import combinations
import math as math

class ensemble(object):
    def __init__(self,np,brownian=False):

        if not isinstance(np, int):
            raise TypeError("The particle number must be int")
        
        self.np = np
        self.init_coord(brownian)

        
    def init_coord(self,br):
        self._v=2.*(uniform.rvs(size=(self.np))-0.5)
        self._c=np.sort(uniform.rvs(size=(self.np)))
        self._mass = np.ones(self.np,dtype=np.float64)
        if(br):
            aux = (self._c[int(self.np/2)+1]-self._c[int(self.np/2)])
            self._c[0] = aux*uniform.rvs(size=(1))+self._c[int(self.np/2)]
            self._mass[0] = 1e1*self._mass[0]
                
    @property
    def v(self,):
        return self._v
            
    @v.setter
    def v(self,value):
        if not isinstance(value,np.ndarray):
            raise TypeError("Invalid velocity")
        if not len(value)==self.np:
            raise TypeError("Invalid number of particles")
        self._v = value

    @property
    def c(self,):
        return self._c
 
    @c.setter
    def c(self,value):
        if not isinstance(value,np.ndarray):
            raise TypeError("Invalid coordinates")
        if not len(value)==self.np:
            raise TypeError("Invalid number of particles")
         
        self._c = value

    @property
    def mass(self,):
        return self._mass
 
    @mass.setter
    def mass(self,value):
        if not isinstance(value,np.ndarray):
            raise TypeError("Invalid coordinates")
        if not len(value)==self.np:
            raise TypeError("Invalid number of particles")
        self._mass = value



class EventManager(ensemble):
    def __init__(self,n,radius=0.01,brownian=False):
        self.n = n
        self._radius = radius
        ensemble.__init__(self,n,brownian) 
        self._couples = np.array([np.arange(self.n-1),np.arange(1,self.n)]).T
        self._nCouples = len(self._couples)
        self._neg = len(np.where(self._c[1:]<self._c[0])[0])
        
    def ghost_times(self,):

        v0 = -self._v[0]
        v1 = self._v[1]
        vn_1 = self._v[self.n-2]
        vn = -self._v[-1]

        c0 = -self._c[0]
        c1 = self._c[1]
        cn_1 = self._c[self.n-2]
        cn = 2.-self._c[-1]

        t0 = (c0-c1)/(v1-v0)
        tn = (cn_1-cn)/(vn-vn_1)

        return t0,tn
        
    def next_event(self,):

#        cp coordinates, velocity

        v_couples = self._v[self._couples]

        c_couples = self._c[self._couples]

        delta_c = c_couples[:,0]-c_couples[:,1]

        delta_v = v_couples[:,1]-v_couples[:,0]
        
        times = np.array([delta_c/delta_v,np.arange(self._nCouples)])

        where_pos = np.where(times[0]>1e-10)[0]

       
        which_couple = times[:,where_pos][:,np.argmin(times[0,where_pos])]

        
        ti,tf = self.ghost_times()
        
        
        if(ti>1e-10 and ti<which_couple[0]):
            which_couple = [ti,0]
        elif(tf>1e-10 and tf<which_couple[0]):
            which_couple = [tf,self._nCouples-1]

        return which_couple

    
    def next_event_BM(self,):
        ''''''''''
        Function giving the next collisional event in
        case of the presence of a Brownian particle.
        The Brownian particle correspond to the index 0
        while the other particles can go through each
        other
        '''''''''

        def get_which_part(ratio):
            
            times = np.array([ratio,np.arange(1,self.n)])
            
            where_pos = np.where(times[0]>1e-10)[0]

            which_particle = [1e6,0]

            try:
                which_particle = times[:,where_pos][:,np.argmin(times[0,where_pos])]
            except:
                which_particle = [1e6,0]

           
            return which_particle
    
                
        distances = (self._c[1:]-self._c[0])
        velocities = (self._v[0]-self._v[1:])
        
        which_particle = get_which_part(distances/velocities) 

        distances = (-self._c[1:]-self._c[0])
        velocities = (self._v[0]+self._v[1:])

        which_particle_l = get_which_part(distances/velocities)

        distances = (2.-self._c[1:]-self._c[0])

        which_particle_r = get_which_part(distances/velocities)

        string = "standard"
        if(which_particle[0]>which_particle_l[0] and which_particle_l[1]<=self._neg):
            string= "left"
            which_particle = which_particle_l.copy()
        if(which_particle[0]>which_particle_r[0] and which_particle_r[1]>self._neg):
            string = "right"
            which_particle = which_particle_r.copy()

      
        return which_particle
            
        
