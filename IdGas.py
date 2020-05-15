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
        self._v=uniform.rvs(size=(self.np))-0.5
        self._c=uniform.rvs(size=(self.np))
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
        self._combs = np.array(list(combinations(np.arange(self.n),2)))
        
    def next_event(self,):

#        cp coordinates, velocity
        def how_long(cp):
            
            return (cp[0,0]-cp[0,1])/(cp[1,1]-cp[1,0])#-2.*self._radius/np.abs(cp[1,1]-cp[1,0])

        couples=np.array([self._c[self._combs],self._v[self._combs]]).swapaxes(0,1)

        times = list(map(how_long,couples))

        time_coup = np.array([times,np.arange(len(self._combs))])

        rem_prev = np.where(time_coup[0]>1e-10)[0]
        

        self._timeCoup = time_coup[:,rem_prev][:,np.argmin(time_coup[0,rem_prev])].copy()

                
        return self._timeCoup[0]

class TimeEvolutor():
    def __init__(self,n,delta=1e-2):
        
        self._delta = delta

        self._em = EventManager(n)

        self._sampling = [[self._em._c.copy(),self._em._v.copy()],]

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

        collision_t = self._em.next_event() + self.timer()

        couple = self._em._combs[int(self._em._timeCoup[1])]

        current_time = self.timer()

        next_dump = (self._lastDump+1)*self._delta

        dumps = int((collision_t-next_dump)/self._delta) 

        # print("############################")
        # print("Dumps %d"%(dumps,))
        # print("Next dump %lf"%(next_dump,))
        # print("Current time %lf"%(current_time,))
        # print("Collision time %lf"%(collision_t,))
        # print("############################")

        if(next_dump<collision_t):
            self.plain_evolution(next_dump-current_time)
            self._sampling.append([self._em._c.copy(),self._em._v.copy()])
            self._lastDump+=1
            
            
            while(dumps>0):
                self.plain_evolution(self._delta)
                self._sampling.append([self._em._c.copy(),self._em._v.copy()])
                self._lastDump+=1
                dumps-=1


            self.plain_evolution(collision_t-self.timer())

        else:

            self.plain_evolution(collision_t-current_time)
            
       # print(self._em._v,self._em._c)
        self.collision(couple)
       # print(self._em._v,self._em._c)

    def evolve_collisions(self,n_col):

        print("Starting time : %f"%(self._time))
        for c in range(n_col):
            self.next_event()
        print("Total time : %f"%(self._time))
        
        
                    

