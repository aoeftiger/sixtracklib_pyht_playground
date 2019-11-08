#!/usr/bin/env python
# coding: utf-8

import sys
assert len(sys.argv) == 1 + 3, 'Require 3 arguments: gpu_device_id, qx, qy'

gpu_device_id, qx, qy = sys.argv[1:]
qx = float(qx)
qy = float(qy)

# In[1]:


import numpy as np

import os
import pickle

import time


# In[2]:


from scipy.constants import e, m_p, c

from scipy.constants import physical_constants


# In[3]:


from cpymad.madx import Madx

import sixtracklib as pyst
import pysixtrack


# In[4]:


###os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device_id
#'2'


# In[5]:


import sys
sys.path.append('/home/oeftiger/gsi/git/python3/PyHEADTAIL/')
from PyHEADTAIL.particles import generators


# In[6]:


nmass = physical_constants['atomic mass constant energy equivalent in MeV'][0] * 1e-3
nmass = 0.931494061 # MAD-X value


# In[7]:


tune_range_qx = np.arange(18.55, 18.95 + 0.01, 0.01)
tune_range_qy = tune_range_qx


# In[8]:






tune_range_qx = np.arange(18.65, 18.95 + 0.01, 0.01)




# # I. Define simulation case

# In[9]:


class Runner(object):
    A = 238 # mass number
    Q = 28 # elementary charges per particle
    Ekin_per_nucleon = 0.2e9 # in eV
    
    epsx_rms_fin = 35e-6 / 4 # geometrical emittances
    epsy_rms_fin = 15e-6 / 4

    limit_n_rms_x = 2
    limit_n_rms_y = 2
    limit_n_rms_z = 3.4

    sig_z = 58 / 4. # in m
    sig_dp = 0.5e-3
    
    def __init__(self, nturns=20000, npart=1000, e_seed=1):
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('error_tables'):
            os.makedirs('error_tables')
        
        self.nturns = nturns
        self.npart = npart
        self.e_seed = e_seed
    
        mass = self.A * nmass * 1e9 * e / c**2 # in kg
        charge = self.Q * e # in Coul

        Ekin = self.Ekin_per_nucleon * self.A
        p0c = np.sqrt(Ekin**2 + 2*Ekin*mass/e * c**2) # in eV

        Etot = np.sqrt(p0c**2 + (mass/e)**2 * c**4) * 1e-9 # in GeV
        p0 = p0c / c * e # in SI units
        gamma = np.sqrt(1 + (p0 / (mass * c))**2)
        beta = np.sqrt(1 - gamma**-2)
        
        self.beta = beta
        self.gamma = gamma
        self.p0 = p0
        self.Etot = Etot
        self.p0c = p0c
        self.charge = charge
        self.mass = mass
        
        epsx_gauss = self.epsx_rms_fin * 1.778
        epsy_gauss = self.epsy_rms_fin * 1.82

        self.epsn_x = epsx_gauss * beta * gamma
        self.epsn_y = epsy_gauss * beta * gamma

        self.beta_z = self.sig_z / self.sig_dp
        
        self.madx = Madx()
        self.madx.options.echo = False
        self.madx.options.warn = False
        self.madx.options.info = False


    def run(self, qx, qy, simulate=False):
        qqx, qqy = int(np.round((qx%1) * 100)), int(np.round((qy%1) * 100))
        
        filename_error_table = "./error_tables/errors_{qqx}_{qqy}_{eseed:d}".format(
            qqx=qqx, qqy=qqy, eseed=self.e_seed)
        
        if os.path.exists('results/' + 
                          os.path.basename(filename_error_table) + '_done'):
            print (f'*** Skipping {qx:.2f}-{qy:.2f} case as *_done file has been found.')
            return

        print ('\n\n\n=== Preparing for Qx = {:.2f} and Qy = {:.2f} ===\n\n\n'.format(qx, qy))
        
        ### SETUP
        twiss = self.setup_madx(self.madx, filename_error_table)
        pyht_beam = self.setup_pyheadtail_particles(twiss)
        pysixtrack_elements = self.setup_pysixtrack(self.madx, filename_error_table)
        trackjob = self.setup_sixtracklib(pysixtrack_elements, pyht_beam)
        
        print ('\n\n\n' + '+'*26 + '\n*** ready for tracking ***\n' + '+'*26 + '\n')
        print ('\n\n\n=== Running at Qx = {:.2f} and Qy = {:.2f} ===\n\n\n'.format(qx, qy))
        
        if simulate:
            trackjob.track_until(0)
        else:
            trackjob.track_until(self.nturns)
        trackjob.collect()
        
        self.teardown(trackjob, filename_error_table, simulate=simulate)
        return trackjob


    def setup_madx(self, madx, filename_error_table):
        madx.call('./SIS100_RF_220618_9slices.thin.seq')

        madx.command.beam(particle='ion', mass=self.A*nmass, 
                          charge=self.Q, energy=self.Etot)

        madx.call('OpticsYEH_BeamParameters.str')
        madx.call('Coll+Errors+BeamDistr.madx')

        madx.input('''
        select, flag=seqedit, class=collimator;
        select, flag=seqedit, class=kicker;
        select, flag=seqedit, class=tkicker;
        select, flag=seqedit, class=elseparator;

        seqedit, sequence=SIS100RING;
            remove, element=selected;
            flatten;
        endedit;

        select, flag=seqedit, class=marker;
        seqedit, sequence=SIS100RING;
            remove, element=selected;
            install, element=SIS100RING$START, s=0;
            flatten;
        endedit;
        ''')

        madx.use(sequence='sis100ring')

        ### --> first match, then add errors, then TWISS!

        madx.input('''
            match, sequence=SIS100RING;
            global, sequence=SIS100RING, q1={qx}, q2={qy};
            vary, name=kqf, step=0.00001;
            vary, name=kqd, step=0.00001;
            lmdif, calls=500, tolerance=1.0e-10;
            endmatch;
        '''.format(qx=qx, qy=qy)
        )

        madx.command.eoption(add=True, seed=1)
        madx.command.exec('EA_EFCOMP_MH()')
        for s in range(1, 10):
            assert madx.command.exec(f'EA_rEFCOMP_QD({s},1)')

        twiss = madx.twiss();

        madx.command.select(flag='error', pattern='QD11..', class_='MULTIPOLE')
        madx.command.select(flag='error', pattern='QD12..', class_='MULTIPOLE')
        madx.command.select(flag='error', pattern='mh1', class_='MULTIPOLE')
        madx.command.select(flag='error', pattern='mh2', class_='MULTIPOLE')
        madx.command.esave(file=filename_error_table)

        madx.input('cavity_voltage = 58.2/1000/number_cavities;')
        
        return twiss

    def setup_pyheadtail_particles(self, twiss):
        # particle initialisation from pyheadtail

        D_x_0 = twiss['dx'][0] * self.beta
        D_y_0 = twiss['dy'][0] * self.beta

        np.random.seed(0)

        pyht_beam = generators.generate_Gaussian6DTwiss(
            self.npart, 1, self.charge, self.mass, twiss['s'][-1], self.gamma,
            twiss['alfx'][0], twiss['alfy'][0], twiss['betx'][0], twiss['bety'][0],
            1, self.epsn_x, self.epsn_y, 1,
            dispersion_x=D_x_0 if D_x_0 else None,
            dispersion_y=D_y_0 if D_y_0 else None,
            limit_n_rms_x=self.limit_n_rms_x**2, 
            limit_n_rms_y=self.limit_n_rms_y**2, 
            limit_n_rms_z=self.limit_n_rms_z**2,
        )

        distribution_z_uncut = generators.gaussian2D(
            self.sig_z**2)
        is_accepted = generators.make_is_accepted_within_n_sigma(
            epsn_rms=self.sig_z,
            limit_n_rms=self.limit_n_rms_z,
        )
        distribution_z_cut = generators.cut_distribution(distribution_z_uncut, is_accepted)

        z, dp = distribution_z_cut(self.npart)
        pyht_beam.z, pyht_beam.dp = z, dp / self.beta_z
        
        return pyht_beam

    def setup_pysixtrack(self, madx, filename_error_table):
        madx.command.readtable(file=filename_error_table, table="errors")
        errors = madx.table.errors

        sis100 = madx.sequence.sis100ring
        
        ### PySixTrack, lattice transfer and preparation!

        pysixtrack_elements, _ = pysixtrack.Line.from_madx_sequence(
            sis100, exact_drift=True, install_apertures=True
        )

        pysixtrack_elements.remove_zero_length_drifts(inplace=True);
        pysixtrack_elements.merge_consecutive_drifts(inplace=True);
        
        # add alignment and multipole errors

        pysixtrack_elements.apply_madx_errors(error_table=errors)
        
        return pysixtrack_elements

    def setup_sixtracklib(self, pysixtrack_elements, pyht_beam):
        ### Load lattice into SixTrackLib

        elements = pyst.Elements.from_line(pysixtrack_elements)
        elements.BeamMonitor(num_stores=self.nturns);

        ### Transfer particles into SixTrackLib

        particles = pyst.Particles.from_ref(
            self.npart, p0c=self.p0c, mass0=self.A*nmass*1e9, q0=self.Q)

        particles.x[:] = pyht_beam.x
        particles.px[:] = pyht_beam.xp
        particles.y[:] = pyht_beam.y
        particles.py[:] = pyht_beam.yp
        particles.zeta[:] = pyht_beam.z
        particles.delta[:] = pyht_beam.dp

        particles.rpp[:] = 1. / (pyht_beam.dp + 1)

        restmass = self.mass * c**2
        restmass_sq = restmass**2
        E0 = np.sqrt((self.p0 * c)**2 + restmass_sq)
        p = self.p0 * (1 + pyht_beam.dp)
        E = np.sqrt((p * c)**2 + restmass_sq)
        particles.psigma[:] = (E - E0) / (self.beta * self.p0 * c)

        gammai = E / restmass
        betai = np.sqrt(1 - 1. / (gammai * gammai))
        particles.rvv[:] = betai / self.beta

        ### prepare trackjob in SixTrackLib

        #trackjob = pyst.CudaTrackJob(elements, particles)
        trackjob = pyst.TrackJob(elements, particles, device=gpu_device_id)

        return trackjob

    def teardown(self, trackjob, filename_error_table, simulate=False):
        store = {}
        filename_error_table = os.path.basename(filename_error_table)

        # statistics
        x = trackjob.output.particles[0].x.reshape((self.nturns, self.npart)).T
        store['std_x'] = np.mean(np.std(x, axis=0)[-50:])
        y = trackjob.output.particles[0].y.reshape((self.nturns, self.npart)).T
        store['std_y'] = np.mean(np.std(y, axis=0)[-50:])

        # losses
        pbuffer = trackjob.particles_buffer.get_object(0)
        if not simulate:
            np.save('results/' + filename_error_table + '_alive.npy', pbuffer.state)
            np.save('results/' + filename_error_table + '_lost_at_element.npy', 
                    pbuffer.at_element[~pbuffer.state.astype(bool)])
            np.save('results/' + filename_error_table + '_lost_at_turn.npy',
                    pbuffer.at_turn[~pbuffer.state.astype(bool)])

        store['losses'] = np.sum(pbuffer.state)

        # finish job
        if not simulate:
            pickle.dump(store, open('results/' + filename_error_table + '_summary.p', 'wb'))

            with open('results/' + filename_error_table + '_done', 'w') as t:
                t.write('')

        self.madx.exit()


# # II. Run the simulation in parameter scan

# In[10]:

#for qx in tune_range_qx:
#    for qy in tune_range_qy:


if __name__ == '__main__':
    simulation = Runner()
    simulation.run(qx, qy, simulate=False)
