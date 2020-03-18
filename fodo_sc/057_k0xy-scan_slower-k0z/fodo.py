#!/usr/bin/env python
# coding: utf-8

import sys
assert len(sys.argv) == 1 + 1, (
    'Require 1 argument: k0xy')

k0x, = sys.argv[1:]
k0x = float(k0x)

Qx = k0x/360.
n_turns = 5 #20000
n_stored_particles = 1000


# In[1]:


# import os, time
# while not os.path.exists('../045_k0xy94deg_based_on_041/done'):
#     time.sleep(1)


# In[2]:


import numpy as np

# In[3]:


from cpymad.madx import Madx

import sixtracklib as pyst


# In[4]:


import h5py


# In[5]:


from scipy.constants import e, m_p, c, epsilon_0


# In[6]:


import sys, os
# sys.path = ["/home/HPC/oeftiger/PyHEADTAIL_py3/python3/PyHEADTAIL/", 
#             "/home/HPC/oeftiger/PyHEADTAIL_py3/"] + sys.path

sys.path = ["/home/oeftiger/gsi/git/python3/PyHEADTAIL/", 
            "/home/oeftiger/gsi/git/"] + sys.path

import pickle


# In[7]:


#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[8]:


from pycuda.autoinit import context
from pycuda import gpuarray as gp

from pycuda.driver import memcpy_dtod_async, Context


# In[9]:


from PyHEADTAIL.general.element import Element
from PyHEADTAIL.general import pmath as pm
from PyHEADTAIL.particles.generators import (
    generate_Gaussian6DTwiss, cut_distribution, make_is_accepted_within_n_sigma)
from PyHEADTAIL.particles import generators

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.spacecharge.pypic_factory import create_3dmesh_from_beam, create_mesh
from PyHEADTAIL.spacecharge.pypic_spacecharge import (
    SpaceChargePIC, SpaceChargePIC_Adaptive25D)

from PyHEADTAIL.monitors.monitors import BunchMonitor, ParticleMonitor

from PyHEADTAIL.general.contextmanager import GPU

from PyHEADTAIL.trackers.rf_bucket import RFBucket


# In[10]:


from PyPIC.GPU.poisson_solver.FFT_solver import (
    GPUFFTPoissonSolver_2_5D, GPUFFTPoissonSolver)
from PyPIC.GPU.pypic import PyPIC_GPU

# not necessary but nice: memory pool sharing between PyHEADTAIL and PyPIC
try:
    from PyHEADTAIL.gpu.gpu_utils import memory_pool
except:
    memory_pool = None


# In[11]:


import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'np.asscalar')


# ### Note by Ingo:
# 
# (09.10.19 18:42) Im Prinzip sind wir bei 40 % bis 5000 Zellen für rms. Die 99.9% saturiert ja relativ bald. Wichtig ist der anfänglich lineare Anstieg. Der wird noch flacher, wenn ich die Gaussverteilung mehr beschneide, oder gar einen transv waterbag nehme. 
# ```
# Referenzwerte:
# k0xy =92 	kxy=80
# k0z=29.5 	kz=29.3
# I=1.3mA 	
# 32 MHz 	
# 10 MeV p 	
# 6d Gaussian 3.4 sig
# length +/- 12 deg (full)
# ```
# 
# (09.10.19 18:56) Der Bunch hat +/- 46 mm Länge. Die normierten rms Emittanzen sind 0.05/0.05/15 mm mrad .
# 
# (10.10.19 13:26) 580 kV je gap und 2 gaps pro FODO-Zelle.
# 
# (10.10.19 15:19) Das wären die Parameter: 
# ```
# Phase-Energy
# Emit [rms] = 0.5408  π.deg.MeV [ Norm. ]
# Emit [100%] = 3.1151  π.deg.MeV [ Norm. ]
# Beta = 25.1831  deg/π.MeV
# Alpha = 0.0000
# β = 0.144844009   γ = 1.010657890
# Mo  = 938.27202900 MeV
# Sigma_Phase [rms] = 3.6905  deg
# Sigma_Energy [rms] = 0.1465  MeV
# ```
# 
# (28.02.20 12:34)  
# 
# ```
# Emit [rms] = 15.0000 π.mm.mrad [ Norm. ]
# β = 0.144843960 γ = 1.010657883
# 
# Sigma_Z [rms] = 13.9107  mm
# 
# Sigma_Dp [rms] = 0.7366  %
# ```

# In[12]:


n_macroparticles = int(4e6)
n_slices_sc = 64
n_sc_nodes = 10

# fixed field map for space charge
n_mesh_nodes = 256 #512
n_mesh_sigma = 24 #24

f_rf = 32e6 # RF frequency in Hz
I = 1.3e-3 * 12 / 24.475 # current in A (I = Q * f_rf)
epsn_x = 0.05e-6 # in [m.rad]
epsn_y = 0.05e-6 # in [m.rad]
sigma_phase = 24 / 4. #3.6905
sigma_dE = 0.1465e6

# epsg_z = 3.13e-05 # in [eV.s]
# sigma_tau = 6.3e-10 # in [s]
# sigma_dp = 1e-15
# rf_voltage = 1e4

# p0c = 100e6 # in eV

Ekin = 10e6 # in eV
charge = e
mass = m_p

circumference = 1

Qy = Qx
Qs = Qx / 30.

###

p0c = np.sqrt(Ekin**2 + 2*Ekin*m_p/e * c**2) # in eV

Etot = np.sqrt(p0c**2 + (m_p/e)**2 * c**4) * 1e-9 # in GeV
p0 = p0c / c * e
gamma = np.sqrt(1 + (p0 / (m_p * c))**2)
beta = np.sqrt(1 - gamma**-2)
epsg_x = epsn_x / (beta * gamma)
epsg_y = epsn_y / (beta * gamma)

h = f_rf / (beta * c / circumference) # ratio of Ingo's 32MHz to our 43MHz h=1 case
sigma_z = sigma_phase / 360. * h / circumference
sigma_dp = sigma_dE / (Etot * 1e9) / beta**2
epsn_z = sigma_z * sigma_dp * 4 * np.pi * p0 / e # sigma_z**2 * 4 * np.pi * p0 / (e * beta_z)
# epsn_z = epsg_z / (beta * gamma)
# bunch_length = 4 * sigma_tau
# sigma_z = beta * c * bunch_length / 4. # in [m]

intensity = I / f_rf / e

eta = -gamma**-2
# Qs = np.sqrt(e * rf_voltage * h * -eta / (2 * np.pi * p0 * beta * c))
rf_voltage = 2 * np.pi * p0 * beta * c * Qs**2 / (e * h * -eta)
beta_z = np.abs(eta) * circumference / (2 * np.pi * Qs)

# beta_z = sigma_z / sigma_dp
# sigma_dp = sigma_z / beta_z


if not os.path.exists('results'):
    os.makedirs('results')

bmon_filename = f'results/k0xy{k0x:.1f}deg_bunchmonitor'
pmon_filename = f'results/k0xy{k0x:.1f}deg_particlemonitor'

if os.path.exists(bmon_filename + '.h5'):
    raise ValueError(
        f'Cannot run for {k0x:.1f}deg: found existing '
        'bunchmonitor in results/ for this case already!')

# In[17]:


def provide_pycuda_array(ptr):
    return gp.GPUArray(n_macroparticles, dtype=np.float64, gpudata=ptr)


# # Set up FODO cell in MAD-X

# In[18]:


madx = Madx()
madx.options.echo = False


# In[19]:


madx.input('''
kqd := -14.17 * 0.1;
kqf := 14.17 * 0.1;
v := 0;

qd: quadrupole, l = 0.05, k1 := kqd / 0.1;   //knl := {0, kqd/2.};
qf: quadrupole, l = 0.1, k1 := kqf / 0.1;         //knl := {0, kqf};
'''
+
'''
rf: rfcavity, volt := v, harmon = {1}, lag = 0;   //, lag = 0.5;
fodo: sequence, l = {0};
qd, at = 0.025;
rf, at = {0} / 4.;
qf, at = {0} / 2.;
rf, at = {0} * 3 / 4.;
qd, at = {0} - 0.025;
endsequence;
'''.format(circumference, h))


# In[20]:


madx.command.beam(particle='proton', energy=str(Etot)) # energy in GeV


# In[21]:


madx.use(sequence='fodo')


# In[22]:


madx.input(
'''match, sequence=fodo;
global, sequence=fodo, q1={Qx}, q2={Qy};
vary, name = kqd, step=0.0001;
vary, name = kqf, step=0.0001;
lmdif, tolerance=1e-20;
endmatch;
'''.format(Qx=Qx, Qy=Qy))


# In[23]:


assert madx.command.select(
    flag='MAKETHIN',
    CLASS='qd',
    SLICE='8',
)
assert madx.command.select(
    flag='MAKETHIN',
    CLASS='qf',
    SLICE='16',
)


# In[24]:


madx.command.makethin(
    makedipedge=False,
    style='teapot',
    sequence='FODO',
)


# # Add space charge placeholders

# In[25]:


madx.input('sc_placeholder: Marker;')


# In[26]:


madx.command.seqedit(sequence='FODO')
for s_sc in np.linspace(start=0, stop=circumference, num=n_sc_nodes, endpoint=False):
    madx.command.install(element='sc_placeholder', at=s_sc)
madx.command.flatten()
madx.command.endedit()


# In[27]:


madx.use(sequence='fodo')


# In[28]:


twiss = madx.twiss()
assert circumference == twiss['s'][-1]


# In[29]:



# # switch on synchrotron motion

# In[30]:


madx.input('v := {}'.format(0.5 * rf_voltage * 1e-6)) # MV --> 0.5x because there are 2 cavities


# # Preparing PyHEADTAIL beam

# In[31]:


D_x_0 = twiss['dx'][0] * beta
D_y_0 = twiss['dy'][0] * beta

np.random.seed(0)

pyht_beam = generate_Gaussian6DTwiss(
    n_macroparticles, intensity, e, m_p, circumference, gamma,
    twiss['alfx'][0], twiss['alfy'][0], twiss['betx'][0], twiss['bety'][0],
    beta_z, epsn_x, epsn_y, epsn_z,
    dispersion_x=D_x_0 if D_x_0 else None,
    dispersion_y=D_y_0 if D_y_0 else None,
    limit_n_rms_x=3.4**2, limit_n_rms_y=3.4**2, limit_n_rms_z=3.4**2,
)


# In[34]:


distribution_z_uncut = generators.gaussian2D(
    sigma_z**2)

is_accepted = generators.make_is_accepted_within_n_sigma(
    epsn_rms=sigma_z,
    limit_n_rms=3.4,
)
distribution_z_cut = generators.cut_distribution(distribution_z_uncut, is_accepted)


# In[35]:


z, dp = distribution_z_cut(n_macroparticles)

pyht_beam.z, pyht_beam.dp = z, dp / beta_z



# # Preparing PySTL for GPU

# In[43]:


pyst_beam = pyst.Particles.from_ref(num_particles=n_macroparticles, p0c=p0c)


# In[44]:


elements = pyst.Elements.from_mad(madx.sequence.FODO, exact_drift=True)


# In[45]:


idx_mad_sc = [i for i, name in enumerate(madx.sequence.FODO.element_names()) 
              if 'sc_placeholder' in name]
sc_optics = {
    'beta_x': twiss['betx'][idx_mad_sc],
    'beta_y': twiss['bety'][idx_mad_sc],
    'D_x': twiss['dx'][idx_mad_sc],
    'D_y': twiss['dy'][idx_mad_sc],
    'x': twiss['x'][idx_mad_sc],
    'y': twiss['y'][idx_mad_sc],
    's': twiss['s'][idx_mad_sc]
}


# In[46]:


trackjob = pyst.CudaTrackJob(elements, pyst_beam)


# # Interface to PyHEADTAIL

# In[47]:


from pycuda import cumath


# In[48]:


class TrackSixTrackLib(Element):
    '''General state.'''
    trackjob = None
    pointers = {}
    context = None
    n_elements = 0

    def __init__(self, trackjob, i_start, i_end, context=context):
        if TrackSixTrackLib.trackjob is None:
            TrackSixTrackLib.trackjob = trackjob

            trackjob.fetch_particle_addresses()
            assert trackjob.last_status_success
            ptr = trackjob.get_particle_addresses() # particleset==0 is default

            TrackSixTrackLib.pointers.update({
                'x': provide_pycuda_array(ptr.contents.x),
                'px': provide_pycuda_array(ptr.contents.px),
                'y': provide_pycuda_array(ptr.contents.y),
                'py': provide_pycuda_array(ptr.contents.py),
                'z': provide_pycuda_array(ptr.contents.zeta),
                'delta': provide_pycuda_array(ptr.contents.delta),
                'rpp': provide_pycuda_array(ptr.contents.rpp),
                'psigma': provide_pycuda_array(ptr.contents.psigma),
                'rvv': provide_pycuda_array(ptr.contents.rvv),
            })
            TrackSixTrackLib.n_elements = len(trackjob.beam_elements_buffer.get_elements())

        self.i_start = i_start
        self.i_end = i_end
        self.is_last_element = (i_end == self.n_elements)

        self.context = context

    def track(self, beam):
        # pass arrays and convert units
        self.pyht_to_stlib(beam)
        # track in SixTrackLib
        trackjob.track_line(self.i_start, self.i_end, 
                            finish_turn=self.is_last_element)
        # to be replaced by barrier:
        trackjob.collectParticlesAddresses()

        assert trackjob.last_track_status_success
        # pass arrays back (converting units back)
        self.stlib_to_pyht(beam)

    def pyht_to_stlib(self, beam):
        self.memcpy(self.pointers['x'], beam.x)
        self.memcpy(self.pointers['px'], beam.xp)
        self.memcpy(self.pointers['y'], beam.y)
        self.memcpy(self.pointers['py'], beam.yp)
        self.memcpy(self.pointers['z'], beam.z)
        self.memcpy(self.pointers['delta'], beam.dp)
        
        # further longitudinal coordinates of SixTrackLib
        rpp = 1. / (beam.dp + 1)
        self.memcpy(self.pointers['rpp'], rpp)

        restmass = beam.mass * c**2
        restmass_sq = restmass**2
        E0 = np.sqrt((beam.p0 * c)**2 + restmass_sq)
        p = beam.p0 * (1 + beam.dp)
        E = cumath.sqrt((p * c) * (p * c) + restmass_sq)
        psigma =  (E - E0) / (beam.beta * beam.p0 * c)
        self.memcpy(self.pointers['psigma'], psigma)
                    
        gamma = E / restmass
        beta = cumath.sqrt(1 - 1. / (gamma * gamma))
        rvv = beta / beam.beta
        self.memcpy(self.pointers['rvv'], rvv)
        
        self.context.synchronize()

    @staticmethod
    def memcpy(dest, src):
        '''device memory copy with pycuda from src GPUArray to dest GPUArray.'''
#         dest[:] = src
#         memcpy_atoa(dest, 0, src, 0, len(src))
        memcpy_dtod_async(dest.gpudata, src.gpudata, src.nbytes)

    def stlib_to_pyht(self, beam):
        beam.x = self.pointers['x']
        beam.xp = self.pointers['px']
        beam.y = self.pointers['y']
        beam.yp = self.pointers['py']
        beam.z = self.pointers['z']
        beam.dp = self.pointers['delta']


# # Prepare PyHEADTAIL style one-turn map

# In[49]:


def rms_beam_size(beta_optics, epsn, disp_optics, 
                  sigma_dp=pyht_beam.sigma_dp(), beta=beta, gamma=gamma):
    return np.sqrt(beta_optics * epsn / (beta * gamma) + (disp_optics * sigma_dp)**2)


# In[50]:


sig_x = rms_beam_size(sc_optics['beta_x'], epsn_x, sc_optics['D_x']).max()
sig_y = rms_beam_size(sc_optics['beta_y'], epsn_y, sc_optics['D_y']).max()


# In[51]:


print ('The smallest horizontal beam size is {:.2f}% smaller than the largest beam size.'.format(
    (sig_x - rms_beam_size(sc_optics['beta_x'], epsn_x, sc_optics['D_x']).min()) / sig_x * 100))

print ('The smallest vertical beam size is {:.2f}% smaller than the largest beam size.'.format(
    (sig_y - rms_beam_size(sc_optics['beta_y'], epsn_y, sc_optics['D_y']).min()) / sig_y * 100))


# In[52]:


slicer_sc = UniformBinSlicer(n_slices_sc, n_sigma_z=4) #z_cuts=slicing_interval)


# In[53]:


slices = pyht_beam.get_slices(slicer_sc)
assert not any(slices.particles_outside_cuts)


# Use the maximum size needed for the transverse grid:

# In[54]:


# mesh_3d = create_3dmesh_from_beam(pyht_beam, [n_mesh_nodes]*2, [n_mesh_sigma]*2, 
#                                   slices=pyht_beam.get_slices(slicer_sc))

mesh_origin = [-n_mesh_sigma * sig_x, 
               -n_mesh_sigma * sig_y]
mesh_distances = [2 * n_mesh_sigma * sig_x / n_mesh_nodes, 
                  2 * n_mesh_sigma * sig_y / n_mesh_nodes]
mesh_3d = create_mesh(mesh_origin, mesh_distances, [n_mesh_nodes]*2, slices=slices)


# In[55]:


# poissonsolver = GPUFFTPoissonSolver_2_5D(mesh_3d, context=context, save_memory=False)
poissonsolver = GPUFFTPoissonSolver(mesh_3d, context=context)
pypic_algorithm = PyPIC_GPU(mesh_3d, poissonsolver, context=context, 
                            memory_pool=memory_pool)


# In[56]:


sum(el.length for el in elements.get_elements() if isinstance(el, pyst.DriftExact))


# In[57]:


one_turn_map = []

relevant_elements = elements.get_elements()

if isinstance(elements.get_elements()[-1], pyst.BeamMonitor):
    relevant_elements = relevant_elements[:-1]

i_last = 0
length_covered = 0
for i_curr, el in enumerate(relevant_elements):
    if not isinstance(el, pyst.DriftExact):
        continue
    length_covered += el.length

    #i_curr == 0 or 
    if el.length != 0 : # only inject SC node at markers (for SC)
        continue

    pyst_node = TrackSixTrackLib(trackjob, i_last, i_curr + 1, context=context)
    one_turn_map.append(pyst_node)

    sc_node = SpaceChargePIC(length_covered, pypic_algorithm)
    one_turn_map.append(sc_node)

    i_last = i_curr
    length_covered = 0

assert pyst_node.i_end == len(relevant_elements)
assert el._offset == relevant_elements[-1]._offset
assert isinstance(el, pyst.DriftExact)
assert el.length == 0

pyst_node.is_last_element = True


# # Tracking space charge and full optics:

# In[58]:


alpha_x = twiss['alfx'][0]
beta_x = twiss['betx'][0]
alpha_y = twiss['alfy'][0]
beta_y = twiss['bety'][0]


# In[59]:


def get_action(u, up, dp=None, alpha_twiss=0, beta_twiss=1, disp=0):
    if disp and dp.any():
        u = u - disp * dp
    gamma_twiss = (1 + alpha_twiss**2) / beta_twiss
    J = 0.5 * (gamma_twiss * u**2 + 2*alpha_twiss * u * up + beta_twiss * up**2)
    return J


# In[60]:


import PyHEADTAIL.gpu.thrust_interface as thrust

def gpu_sort(ary):
    sortd = ary.copy() # thrust sorts in place
    thrust.sort_double(sortd)
    return sortd

pm._CPU_numpy_func_dict['sort'] = np.sort
pm._GPU_func_dict['sort'] = gpu_sort

pm.update_active_dict(pm._CPU_numpy_func_dict)


# In[61]:


J_quantiles = np.arange(0, 1, 0.05) + 0.05

J_quantiles_ids = list((J_quantiles * n_macroparticles).astype(int) - 1)

def get_J_xy_quantiles(
        pyht_beam, J_quantiles_ids=J_quantiles_ids, 
        alpha_x=alpha_x, alpha_y=alpha_y, 
        beta_x=beta_x, beta_y=beta_y):    
    J_x = get_action(pyht_beam.x, pyht_beam.xp, alpha_twiss=alpha_x, beta_twiss=beta_x)
    J_x_quantiles = pm.ensure_CPU(pm.sort(J_x))[J_quantiles_ids]

    J_y = get_action(pyht_beam.y, pyht_beam.yp, alpha_twiss=alpha_y, beta_twiss=beta_y)
    J_y_quantiles = pm.ensure_CPU(pm.sort(J_y))[J_quantiles_ids]
    return J_x_quantiles, J_y_quantiles


# In[62]:


bunchmon = BunchMonitor(bmon_filename, n_turns + 1, write_buffer_every=1)
partmon = ParticleMonitor(pmon_filename, stride=n_macroparticles // n_stored_particles)

# emittance quantiles evolution
rec_epsn_q_x = np.empty((n_turns + 1, len(J_quantiles)), dtype=float)
rec_epsn_q_y = np.empty_like(rec_epsn_q_x)

J_x, J_y = get_J_xy_quantiles(pyht_beam)
rec_epsn_q_x[0, :] = beta * gamma * J_x
rec_epsn_q_y[0, :] = beta * gamma * J_y

with GPU(pyht_beam):
    bunchmon.dump(pyht_beam)
    pyht_beam.sort_for('id')
    partmon.dump(pyht_beam)

    for i in range(1, n_turns+1):
        for m in one_turn_map:
            m.track(pyht_beam)

        bunchmon.dump(pyht_beam)

        pyht_beam.sort_for('id')
        partmon.dump(pyht_beam)

        J_x, J_y = get_J_xy_quantiles(pyht_beam)
        rec_epsn_q_x[i, :] = beta * gamma * J_x
        rec_epsn_q_y[i, :] = beta * gamma * J_y

        sys.stdout.write('\rTurn {}/{}'.format(i, n_turns))


# # Save simulation results

# In[63]:


pickle.dump(pyht_beam, open(f'results/k0xy{k0x:.1f}deg_beam.p', 'wb'))


# In[64]:


np.save(f'results/k0xy{k0x:.1f}deg_epsn_quantiles_x', rec_epsn_q_x)
np.save(f'results/k0xy{k0x:.1f}deg_epsn_quantiles_y', rec_epsn_q_y)


# # Basic consistency checks

# ## No particles left the grid:

# Cross-checking the amplitudes of the particles, ensuring that they remain within what the transverse PIC grid covers:

# In[71]:


grid_x_action = get_action(sc_node.pypic.mesh.x0, 0, beta_twiss=beta_x)


# In[72]:


J_x = get_action(pyht_beam.x, pyht_beam.xp, alpha_twiss=alpha_x, beta_twiss=beta_x)
id_max = np.argmax(J_x)

assert J_x[id_max] < grid_x_action, 'largest horizontal amplitude particle is off the grid!'


# In[73]:


grid_y_action = get_action(sc_node.pypic.mesh.y0, 0, beta_twiss=beta_y)


# In[74]:


J_y = get_action(pyht_beam.y, pyht_beam.yp, alpha_twiss=alpha_y, beta_twiss=beta_y)
id_max = np.argmax(J_y)

assert J_y[id_max] < grid_y_action, 'largest vertical amplitude particle is off the grid!'


# ## All particles alive:

# No particles met the 1m global aperture of SixTrackLib:

# In[75]:


all(pyst_beam.state)


# Numerical cross-check: no particles acquired $\equiv 0$ coordinate or momentum values (all remain finite):

# In[76]:


assert all([all(pyht_beam.x), all(pyht_beam.xp), all(pyht_beam.y), all(pyht_beam.yp), all(pyht_beam.z)])

print (f'\n\n\nSimulation finished for k0xy={k0x:.1f}deg!\n\n')
