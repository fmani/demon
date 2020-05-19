import numpy as np
from scipy.stats import uniform
from itertools import combinations
import math as math

class ensemble(object):
    def __init__(self,np):

        if not isinstance(np, int):
            raise TypeError("The particle number must be int")
        
        self.np = np
        self.init_coord()
        
        

    def init_coord(self,):
        self._v=2.*(uniform.rvs(size=(self.np))-0.5)
        self._c=np.sort(uniform.rvs(size=(self.np)))
        self._mass = np.ones(self.np,dtype=np.float64)

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
    def __init__(self,n,radius=0.01):
        self.n = n
        self._radius = radius
        ensemble.__init__(self,n) 
        self._couples = np.array([np.arange(self.n-1),np.arange(1,self.n)]).T
        self._nCouples = len(self._couples)

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

class TimeEvolutor():
    def __init__(self,n,delta=1e-2):

        self._delta = delta

        self._em = EventManager(n)

        self._sampling = [[self._em._c.copy(),self._em._v.copy()],]

        self._energy=[self.energy(),]
        
        self._L = 1.

        self._time = 0.
        
        self._lastDump = 0.


    def timer(self,):
        return self._time
        
    def plain_evolution(self, delta_t):
        
        tent_c = self._em._c + delta_t*self._em._v
        
        to_change_neg = np.where(tent_c<0)[0]

        tent_c[to_change_neg] = np.abs(tent_c[to_change_neg])

        to_change_pos = np.where(tent_c>1)[0]

        tent_c[to_change_pos] = 2-tent_c[to_change_pos]

        self._em._c = np.copy(tent_c)
        
        self._em._v[to_change_pos] *= -1.

        self._em._v[to_change_neg] *= -1.

        self._time += delta_t


    def dump(self,):
        self._energy.append(self.energy())
        self._sampling.append([self._em._c.copy(),self._em._v.copy()])
        self._lastDump+=1
            
        
        
    def collision(self, couple):

        masses = self._em._mass[couple]
        
        def m_matrix(m):

             sum_m = np.sum(m)
             
             diff_m = m[1]-m[0]

             r1 = [-diff_m/sum_m,2.*m[1]/sum_m]

             r2 = [2.*m[0]/sum_m,diff_m/sum_m]

             m_matr = np.matrix([r1,r2])
             
             return m_matr
             

        m = m_matrix(masses)

        vels = m*np.matrix(self._em._v[couple]).T
        
        self._em._v[couple] = np.copy(np.array(vels).squeeze())
        
        
    def next_event(self,):
        
        couple = self._em.next_event()
       
        collision_t = couple[0] + self.timer()

        couple = self._em._couples[int(couple[1])]

        current_time = self.timer()

        next_dump = (self._lastDump+1)*self._delta

        dumps = int((collision_t-next_dump)/self._delta) 

        
        if(next_dump<collision_t):
            self.plain_evolution(next_dump-current_time)
            self.dump()
                       
            
            while(dumps>0):
                self.plain_evolution(self._delta)
                self.dump()
                dumps-=1


            self.plain_evolution(collision_t-self.timer())

        else:

            self.plain_evolution(collision_t-current_time)


        self.collision(couple)


    def evolve_collisions(self,n_col):

        print("Starting time : %f"%(self._time))
        for c in range(n_col):
            self.next_event()
        print("Total time : %f"%(self._time))
        
        
    @property
    def mass(self,):
        return self._em._mass
 
    @mass.setter
    def mass(self,value):
        if not isinstance(value,np.ndarray):
            raise TypeError("Invalid coordinates")
        if not len(value)==self._em.n:
            raise TypeError("Invalid number of particles")
        self._em._mass = value


    @property
    def v(self,):
        return self._em._v
 
    @v.setter
    def v(self,value):
        if not isinstance(value,np.ndarray):
            raise TypeError("Invalid coordinates")
        if not len(value)==self._em.n:
            raise TypeError("Invalid number of particles")
        self._em._v = value
        self._sampling[0]=[self._em._c.copy(),self._em._v.copy()]
        self._energy[0]=self.energy()

    def energy(self,):
        tmp = self._em._v**2
        return np.average(0.5*self._em._mass*tmp)
