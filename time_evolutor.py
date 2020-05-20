from IdGas import ensemble
from IdGas import EventManager
import numpy as np
from scipy.stats import uniform


class TimeEvolutor():
    def __init__(self,n,delta=1e-2,brownian=False):

        self._delta = delta

        self._brownian = brownian
        
        self._em = EventManager(n,brownian=self._brownian)

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


    def check_positions(self,):
        ev_c = self._em._c + self._em._v*1e-10
        neg = len(np.where(ev_c[1:]<ev_c[0])[0])
        if(neg!=self._em._neg):
            print(self._em._c[1:][np.where((ev_c[1:]-1e-3)<ev_c[0])[0]])
            print(self._em._c,self._em._neg)
            print("Unwanted mixing occurred at time %lf"%(self.timer(),))
            #raise ValueError("Unwanted mixing occurred at time %lf"%(self.timer(),))
        
        
    def next_event(self,):

        if not self._brownian:
            couple = self._em.next_event()
        else:
            couple = self._em.next_event_BM()

        
        collision_t = couple[0] + self.timer()

        if not self._brownian:
            couple = self._em._couples[int(couple[1])]
        else:
            couple = [0,int(couple[1])]
            
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
        self.check_positions()
        
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
