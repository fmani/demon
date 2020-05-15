#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from style import *
from scipy.special import gamma
from mpl_toolkits import mplot3d
from scipy import stats  
from matplotlib.animation import FuncAnimation
import pickle as pk


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


cols=colors()


# In[6]:


from IdGas import *


# In[105]:


N_PARTS = 100#1000
N_COL = 5#500000


# In[106]:


prova=TimeEvolutor(N_PARTS,1e-3)

prova.mass= uniform.rvs(size=(N_PARTS))
prova.v= np.concatenate((np.ones(int(N_PARTS/2))*0.5,-np.ones(int(N_PARTS/2))*0.5))

prova.evolve_collisions(N_COL)

sample = np.array(prova._sampling)

n_spl = len(sample)

pk.dump(sample,open("./sample.pk","wb"))


# In[103]:


weights = np.ones_like(sample[0,1,:])/len(sample[0,1,:])
kw_parts = {'marker':'.','linestyle':'','markersize':1}

lnspc=np.linspace(-4,4,400)
for i in np.arange(n_spl)[4000:4001]:
    fig, ax,ax2 = canvas_particles()
#     ax.plot(sample[i,0,::2],np.zeros(int(N_PARTS/2)),color=cols[0],**kw_parts)
#     ax.plot(sample[i,0,1::2],np.zeros(int(N_PARTS/2)),color=cols[1],**kw_parts)
    ax.vlines(0,-0.05,0.05)
    ax.vlines(1,-0.05,0.05)
    m, s = stats.norm.fit(sample[i,1,:])
    print(m,s)
    pdf_g = stats.norm.pdf(lnspc, m, s)
    s=0.4
    ax2.plot(lnspc,(1/np.sqrt(2*np.pi*s))*np.exp(-lnspc**2/(2*s**2)))
    ax2.plot(lnspc, pdf_g,linestyle='-.',color=cols[0])
    #plt.xlim(-2,2)
    ax2.hist(sample[i,1,:],range=(-4,4),bins=100,density=True,alpha=0.7, rwidth=0.85,facecolor=cols[1])
    #ax2.set_ylim(-4,4)
    fig.savefig('./video_img/particle_pos_%d.jpg'%(i,),format='jpg',bbox_inches='tight',facecolor='white',dpi=200)
    #plt.close()
#plt.show()    


# In[33]:


# lnspc=np.linspace(-10,10,100)
# pdf_g = stats.maxwell.pdf(lnspc, 0,0.4)
# plt.plot(lnspc,pdf_g)

