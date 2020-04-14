#!/usr/bin/env python
# coding: utf-8

import sys
assert len(sys.argv) == 1 + 4, (
    'Require 4 arguments: GPU ID, qx, qy, e_seed')

gpu_device_id, qx, qy, e_seed = sys.argv[1:]
qx = float(qx)
qy = float(qy)
e_seed = int(e_seed)

# In[1]:


import numpy as np

import os
import pickle

import time

import h5py

# In[2]:




# In[3]:


from scipy.constants import e, m_p, c, epsilon_0
from scipy.constants import physical_constants


# In[4]:


from cpymad.madx import Madx

import sixtracklib as pyst
import pysixtrack
import pysixtrack.be_beamfields.tools as bt

# In[4]:


os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device_id
#'2'

import sys
sys.path.append('/home/oeftiger/gsi/git/python3/PyHEADTAIL/')
sys.path.append('/home/oeftiger/gsi/git/')



sys.path = [
    "/home/HPC/oeftiger/PyHEADTAIL_py3/python3/PyHEADTAIL/",
    "/home/HPC/oeftiger/PyHEADTAIL_py3/",
] + sys.path

from pycuda.autoinit import context
from pycuda import gpuarray as gp
from pycuda import cumath

from pycuda.driver import memcpy_dtod_async
##
from PyHEADTAIL.general.element import Element
from PyHEADTAIL.general import pmath as pm
from PyHEADTAIL.particles import generators

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.spacecharge.pypic_factory import (
    create_3dmesh_from_beam, create_mesh)
from PyHEADTAIL.spacecharge.pypic_spacecharge import (
    SpaceChargePIC, SpaceChargePIC_Adaptive25D)
from PyHEADTAIL.spacecharge.spacecharge import (
    TransverseGaussianSpaceCharge)

from PyHEADTAIL.aperture.aperture import Aperture

from PyHEADTAIL.monitors.monitors import BunchMonitor, ParticleMonitor

from PyHEADTAIL.general.contextmanager import GPU
##
from PyPIC.GPU.poisson_solver.FFT_solver import (
    GPUFFTPoissonSolver_2_5D, GPUFFTPoissonSolver)
from PyPIC.GPU.pypic import PyPIC_GPU

# not necessary but nice: memory pool sharing between PyHEADTAIL and PyPIC
try:
    from PyHEADTAIL.gpu.gpu_utils import memory_pool
except:
    memory_pool = None

# In[6]:

from analytic_sc import AnalyticTransverseGaussianSC

nmass = physical_constants['atomic mass constant energy equivalent in MeV'][0] * 1e-3
# nmass = 0.931494061 # MAD-X value


# In[7]:


# hack replacing SpaceChargeBunched with simple drifts of length int(0),
# used for indicating space charge positions in the lattice
from pysixtrack.elements import DriftExact
class myDriftExact(DriftExact):
    def __new__(cls, *args, **kwargs):
        return DriftExact.__new__(DriftExact, length=0.)
pysixtrack.elements.SpaceChargeBunched = myDriftExact




#tune_range_qx = np.arange(18.55, 18.95 + 0.01, 0.01)
#tune_range_qy = tune_range_qx
#e_seed = 3


with_errors = True #False
with_SC = True
force_zero = True # centroid forced to zero orbit offset!
with_warm_quads = True
install_apertures = True
n_macroparticles = 5000 #int(1e6) # used in provide_pycuda_array

#err_file = 'Coll+Errors+BeamDistr.madx'
err_file = 'newColl+Errors+BeamDistr_onlyk1l.madx'

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
    limit_n_rms_z = 2

    sig_z = 58 / 4. * 0.3/0.33
    sig_dp = 0.5e-3 * 0.3/0.33

    intensity = 0.625e11

    # fixed grid for PIC space charge
    n_slices_sc = 64
    n_mesh_nodes = 128
    n_mesh_sigma = 12
    n_scnodes = 501

    n_stored_particles = 1000

    def __init__(self, nturns=20000, npart=n_macroparticles, e_seed=e_seed):
        if not os.path.exists('results'):
            os.makedirs('results')
        if with_errors and not os.path.exists('error_tables'):
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

        epsx_gauss = self.epsx_rms_fin * 1.43
        epsy_gauss = self.epsy_rms_fin * 1.41

        self.epsn_x = epsx_gauss * beta * gamma
        self.epsn_y = epsy_gauss * beta * gamma

        self.sig_z = self.sig_z #* 1.22
        self.sig_dp = self.sig_dp #* 1.22

        self.beta_z = self.sig_z / self.sig_dp

        self.madx = Madx()
        self.madx.options.echo = False
        self.madx.options.warn = False
        self.madx.options.info = False

    def run(self, qx, qy, fake=False, with_errors=with_errors, with_SC=with_SC, with_warm_quads=with_warm_quads):
        qqx, qqy = int(np.round((qx%1) * 100)), int(np.round((qy%1) * 100))

        filename_error_table = (
            "./error_tables/errors_{qqx}_{qqy}_{eseed:d}".format(
                qqx=qqx, qqy=qqy, eseed=self.e_seed))

        if os.path.exists('results/' +
                          os.path.basename(filename_error_table) + '_done'):
            print (f'*** Skipping {qx:.2f}-{qy:.2f} case as *_done file'
                   ' has been found.')
            return

        print ('\n\n\n=== Preparing for Qx = {:.2f} and '
               'Qy = {:.2f} and e_seed = {:d} ===\n\n\n'.format(
                    qx, qy, self.e_seed))

        ### SETUP
        if with_warm_quads:
            seqname = 'sis100ring'
        else:
            seqname = 'sis100cold'

        twiss = self.setup_madx(self.madx, filename_error_table, with_errors, with_warm_quads)
        sis100 = getattr(self.madx.sequence, seqname)

        if with_SC:
            sc_lengths = self.add_madx_sc_markers(
                self.madx, sis100, seqname)

        pysixtrack_elements = self.setup_pysixtrack(
            self.madx, filename_error_table, with_errors, with_SC, with_warm_quads)
        if with_SC:
            closed_orbit = self.correct_closed_orbit(pysixtrack_elements, force_zero=force_zero)
        else:
            closed_orbit = None
        pyht_beam = self.setup_pyheadtail_particles(twiss, closed_orbit)
        add_STL_attrs_to_PyHT_beam(pyht_beam) # for loss rearrangement in PyHT
        cudatrackjob = self.setup_sixtracklib(pysixtrack_elements, pyht_beam)
        one_turn_map = self.setup_pyheadtail_map(
            self.madx, cudatrackjob, pyht_beam, pysixtrack_elements, sc_lengths, seqname)

        print ('\n\n\n' + '+'*26 + '\n*** ready for tracking ***\n' +
               '+'*26 + '\n')
        print ('\n\n\n=== Running at Qx = {:.2f} and '
               'Qy = {:.2f} and e_seed = {:d} ===\n\n\n'.format(
                    qx, qy, self.e_seed))

        if fake:
            bmon_name, pmon_name = self.track(pyht_beam, one_turn_map, nturns=0)
        else:
            bmon_name, pmon_name = self.track(pyht_beam, one_turn_map)
        # trackjob.collect()

        self.teardown(cudatrackjob, filename_error_table, bmon_name,
                      pmon_name, fake=fake)
        return cudatrackjob


    def setup_madx(self, madx, filename_error_table, with_errors=False, with_warm_quads=True):
        #madx.call('./SIS100_RF_220618_9slices.thin.seq')
        #madx.call('./SIS100_RF_220618_9slices_AllCold.thin.seq')
        #madx.call('./SIS100RING_cold.seq')
        if with_warm_quads:
            madx.call('./SIS100RING_220618.seq')
            k1nl_s52qd11_factor = 1.0139780
            k1nl_s52qd12_factor = 1.0384325
            seqname = 'sis100ring'
        else:
            madx.call('./SIS100RING_220618_AllCold.seq')
            k1nl_s52qd11_factor = 1.0
            k1nl_s52qd12_factor = 1.0
            seqname = 'sis100cold'

        madx.command.beam(particle='ion', mass=self.A*nmass,
                          charge=self.Q, energy=self.Etot)

        madx.input(f'''
            kqd := -0.2798446835;
            kqf := 0.2809756135;

            K1NL_S00QD1D :=  kqd ;
            K1NL_S00QD1F :=  kqf ;
            K1NL_S00QD2F :=  kqf ;
            K1NL_S52QD11 :=  {k1nl_s52qd11_factor}   *   kqd ;
            K1NL_S52QD12 :=  {k1nl_s52qd11_factor}   *   kqf ;
        ''')

        assert madx.input('''
            select, flag=seqedit, class=collimator;
            select, flag=seqedit, class=hmonitor;
            select, flag=seqedit, class=vmonitor;
            select, flag=seqedit, class=kicker;
            select, flag=seqedit, class=tkicker;

            seqedit, sequence={sn};
                remove, element=selected;
                flatten;
            endedit;

            select, flag=seqedit, class=marker;
            seqedit, sequence={sn};
                remove, element=selected;
                install, element={sn}$START, s=0;
                flatten;
            endedit;
        '''.format(sn=seqname))

        madx.use(sequence=seqname)

        assert madx.command.select(
            flag='MAKETHIN',
            class_='QUADRUPOLE',
            slice_='9',
        )

        assert madx.command.select(
            flag='MAKETHIN',
            class_='SBEND',
            slice_='9',
        )

        assert madx.command.makethin(
            makedipedge=True,
            style='teapot',
            sequence=seqname,
        )

        madx.call('OpticsYEH_BeamParameters.str')
        madx.call(err_file)

        madx.use(sequence=seqname)

        ### --> first match, then add errors, then TWISS!

        madx.input('''
            match, sequence={sn};
            global, sequence={sn}, q1={qx}, q2={qy};
            vary, name=kqf, step=0.00001;
            vary, name=kqd, step=0.00001;
            lmdif, calls=500, tolerance=1.0e-10;
            endmatch;
        '''.format(qx=qx, qy=qy, sn=seqname)
        )

        if with_errors:
            e_seed = self.e_seed
            #madx.command.eoption(add=True, seed=e_seed)
            #madx.command.exec('EA_EFCOMP_MH()')
            for s in range(1, 10):
                assert madx.command.exec(f'EA_rEFCOMP_MH({s}, {e_seed})')
                assert madx.command.exec(f'EA_rEFCOMP_QD({s}, {e_seed})')

        twiss = madx.twiss();

        if with_errors:
            madx.command.select(flag='error', pattern='QD11', class_='multipole')
            madx.command.select(flag='error', pattern='QD12', class_='multipole')
            madx.command.select(flag='error', pattern='MH1', class_='multipole')
            madx.command.select(flag='error', pattern='MH2', class_='multipole')
            madx.command.esave(file=filename_error_table)

        madx.input('cavity_voltage = 58.2/1000/number_cavities;')

        return twiss

    def setup_pyheadtail_particles(self, twiss, closed_orbit):
        # particle initialisation from pyheadtail

        if closed_orbit is not None:
            x_co = twiss['x'][0]
            y_co = twiss['y'][0]
        else:
            x_co = 0
            y_co = 0

        D_x_0 = twiss['dx'][0] * self.beta
        D_y_0 = twiss['dy'][0] * self.beta

        Dp_x_0 = twiss['dpx'][0] * self.beta
        Dp_y_0 = twiss['dpy'][0] * self.beta

        np.random.seed(0)

        pyht_beam = generators.generate_Gaussian6DTwiss(
            self.npart, self.intensity, self.charge, self.mass,
            twiss['s'][-1], self.gamma,
            twiss['alfx'][0], twiss['alfy'][0],
            twiss['betx'][0], twiss['bety'][0],
            1, self.epsn_x, self.epsn_y, 1,
            dispersion_x=None, #D_x_0 if D_x_0 else None,
            dispersion_y=None, #D_y_0 if D_y_0 else None,
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
        distribution_z_cut = generators.cut_distribution(
            distribution_z_uncut, is_accepted)

        z, dp = distribution_z_cut(self.npart)
        pyht_beam.z, pyht_beam.dp = z, dp / self.beta_z

        # recentre on 0 to avoid dipolar motion:
        pyht_beam.x -= pyht_beam.mean_x()
        pyht_beam.xp -= pyht_beam.mean_xp()
        pyht_beam.y -= pyht_beam.mean_y()
        pyht_beam.yp -= pyht_beam.mean_yp()
        pyht_beam.z -= pyht_beam.mean_z()
        pyht_beam.dp -= pyht_beam.mean_dp()

        # PyHT generates around 0, need to offset with closed orbit:
        pyht_beam.x += x_co
        pyht_beam.y += y_co
        # add dispersive contribution to coordinates:
        pyht_beam.x += D_x_0 * pyht_beam.dp
        pyht_beam.y += D_y_0 * pyht_beam.dp
        # also need to add D'_{x,y} to momenta:
        pyht_beam.xp += Dp_x_0 * pyht_beam.dp
        pyht_beam.yp += Dp_y_0 * pyht_beam.dp

        return pyht_beam

    def setup_pysixtrack(
            self, madx, filename_error_table, with_errors, with_SC, with_warm_quads):
        if with_warm_quads:
            seqname = 'sis100ring'
        else:
            seqname = 'sis100cold'
        if with_errors:
            madx.command.readtable(file=filename_error_table, table="errors")
            errors = madx.table.errors

        sis100 = getattr(madx.sequence, seqname)

        ### PySixTrack, lattice transfer and preparation!

        pysixtrack_elements = pysixtrack.Line.from_madx_sequence(
            sis100, exact_drift=True, install_apertures=install_apertures
        )

        # pysixtrack_elements.remove_zero_length_drifts(inplace=True)
        # pysixtrack_elements.merge_consecutive_drifts(inplace=True)

        # add alignment and multipole errors

        if with_errors:
            pysixtrack_elements.apply_madx_errors(error_table=errors)

        return pysixtrack_elements

    def add_madx_sc_markers(self, madx, madx_sequence, seqname):
        twiss = madx.twiss()

        tmp_elements = pysixtrack.Line.from_madx_sequence(madx_sequence)

        l_target = twiss.summary.length / self.n_scnodes
        l_fuzz = l_target / 2.

        sc_locations, sc_lengths = bt.determine_sc_locations(
            tmp_elements, self.n_scnodes, l_fuzz
        )

        sc_names = ["sc%d" % number for number in range(len(sc_locations))]

        bt.install_sc_placeholders(
            madx, seqname, sc_names, sc_locations, mode="Bunched")

        return sc_lengths

    # def add_pysixtrack_sc_nodes(
    #         self, madx, pysixtrack_elements, sc_lengths):
    #     pst_elements = pysixtrack_elements

        # mad_sc_names, sc_twdata = bt.get_spacecharge_names_twdata(
        #     madx, "SIS100RING", mode=self.sc_mode
        # )

        # # Check consistency
        # if self.sc_mode == "Bunched":
        #     sc_elements, sc_names = pst_elements.get_elements_of_type(
        #         pysixtrack.elements.SpaceChargeBunched
        #     )
        # elif self.sc_mode == "Coasting":
        #     sc_elements, sc_names = pst_elements.get_elements_of_type(
        #         pysixtrack.elements.SpaceChargeCoasting
        #     )
        # else:
        #     raise ValueError("SC mode not understood")

        # bt.check_spacecharge_consistency(
        #     sc_elements, sc_names, sc_lengths, mad_sc_names
        # )

        # # Setup spacecharge in the line
        # if self.sc_mode == "Bunched":
        #     bt.setup_spacecharge_bunched_in_line(
        #         sc_elements,
        #         sc_lengths,
        #         sc_twdata,
        #         self.beta * self.gamma,
        #         self.intensity,
        #         self.sig_z/1.22,
        #         self.sig_dp /1.22,
        #         35e-6/4 * self.beta * self.gamma,
        #         15e-6/4 * self.beta * self.gamma,
        #     )

        # elif self.sc_mode == "Coasting":
        #     raise NotImplementedError(
        #         'have to supply line_density')
        #     bt.setup_spacecharge_coasting_in_line(
        #         sc_elements,
        #         sc_lengths,
        #         sc_twdata,
        #         self.beta * self.gamma,
        #         line_density,
        #         self.sig_dp,
        #         self.epsn_x,
        #         self.epsn_y,
        #     )

    def correct_closed_orbit(self, pysixtrack_elements, rtol=5e-10, force_zero=False):
        print ('\n\n*** Finding and correcting closed orbit offset '
               'for fixed frozen space nodes.\n\n')

        if force_zero:
            return pysixtrack.Particles(
                p0c=self.p0c, mass0=self.A*nmass*1e9, q0=self.Q)

        # need to have RF voltage at zero for orbit..
        rf_voltages = []
        for el in pysixtrack_elements.elements:
            if isinstance(el, pysixtrack.elements.SpaceChargeBunched):
                el.enabled = False
            if isinstance(el, pysixtrack.elements.Cavity):
                rf_voltages.append(el.voltage)
                el.voltage = 0

        p_co = pysixtrack_elements.find_closed_orbit(self.p0c)
        co = pysixtrack_elements.track_elem_by_elem(p_co.copy())

        for el, local_co in zip(pysixtrack_elements.elements, co):
            if isinstance(el, pysixtrack.elements.SpaceChargeBunched):
                el.x_co = local_co.x
                el.y_co = local_co.y

                el.enabled = True

        co = pysixtrack_elements.track_elem_by_elem(p_co)

        assert np.abs(co[0].x - co[-1].x) <= rtol * np.abs(co[0].x)
        assert np.abs(co[0].y - co[-1].y) <= rtol * np.abs(co[0].y)

        # set back RF voltages
        for el in pysixtrack_elements.elements:
            if isinstance(el, pysixtrack.elements.Cavity):
                el.voltage = rf_voltages.pop(0)

        return p_co

    def setup_sixtracklib(self, pysixtrack_elements, pyht_beam):
        ### Load lattice into SixTrackLib

        elements = pyst.Elements.from_line(pysixtrack_elements)
        # elements.BeamMonitor(num_stores=self.nturns);

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

        particles.particle_id[:] = pyht_beam.id

        ### prepare trackjob in SixTrackLib

        trackjob = pyst.CudaTrackJob(elements, particles)
        # trackjob = pyst.TrackJob(elements, particles,
        #                          device=gpu_device_id)

        return trackjob

    def setup_pyheadtail_map(self, madx, cudatrackjob, pyht_beam,
                             pysixtrack_elements, sc_lengths, seqname):
        assert with_SC
        mad_sc_names, sc_twdata = bt.get_spacecharge_names_twdata(
            madx, seqname, mode='Bunched'
        )
        sc_sig_x = rms_beam_size(
            np.asarray(sc_twdata['betx']), self.epsn_x,
            np.asarray(sc_twdata['dispersion_x']),
            self.sig_dp, self.beta, self.gamma)
        sc_sig_y = rms_beam_size(
            np.asarray(sc_twdata['bety']), self.epsn_y,
            np.asarray(sc_twdata['dispersion_y']),
            self.sig_dp, self.beta, self.gamma)
        print ('The smallest horizontal beam size is {:.2f}% smaller '
               'than the largest beam size.'.format(
            (sc_sig_x.max() - sc_sig_x.min()) / sc_sig_x.max() * 100))
        print ('The smallest vertical beam size is {:.2f}% smaller '
               'than the largest beam size.'.format(
            (sc_sig_y.max() - sc_sig_y.min()) / sc_sig_y.max() * 100))
        sig_x = sc_sig_x.max()
        sig_y = sc_sig_y.max()

        #slicer_sc = UniformBinSlicer(self.n_slices_sc,
        #                             n_sigma_z=1.5 * self.limit_n_rms_z)
        #slices = pyht_beam.get_slices(slicer_sc)
        #assert not any(slices.particles_outside_cuts)

        #n_mesh_sigma = self.n_mesh_sigma
        #n_mesh_nodes = self.n_mesh_nodes
        #mesh_origin = [-n_mesh_sigma * sig_x,
        #               -n_mesh_sigma * sig_y]
        #mesh_distances = [2 * n_mesh_sigma * sig_x / n_mesh_nodes,
        #                  2 * n_mesh_sigma * sig_y / n_mesh_nodes]
        #mesh_3d = create_mesh(mesh_origin, mesh_distances, [n_mesh_nodes]*2,
        #                      slices=slices)

        #poissonsolver = GPUFFTPoissonSolver_2_5D(mesh_3d, context=context,
        #                                         save_memory=False)
        ## poissonsolver = GPUFFTPoissonSolver(mesh_3d, context=context)
        #pypic_algorithm = PyPIC_GPU(mesh_3d, poissonsolver, context=context,
        #                            memory_pool=memory_pool)

        # assert np.isclose(twiss.summary.length, sum(
        #         el.length for el in elements.get_elements()
        #         if isinstance(el, pyst.DriftExact))), \
        #     'SixTrackLib lattice element lengths do not sum up to circumference'

        sc_sig_x = gp.to_gpu(sc_sig_x)
        sc_sig_y = gp.to_gpu(sc_sig_y)

        one_turn_map = []

        ids_sc = [i for i, el in enumerate(pysixtrack_elements.elements)
                  if hasattr(el, 'length') and el.length is 0]

        relevant_elements = pysixtrack_elements.elements
        n_relevant_elements = len(relevant_elements)
        # if isinstance(elements.get_elements()[-1], pyst.BeamMonitor):
        #     n_relevant_elements += 1

        i_last = 0
        for i_curr, el in enumerate(relevant_elements):
            if not i_curr in ids_sc:
                continue

            i_sc = list(ids_sc).index(i_curr)
            length_covered = sc_lengths[i_sc]

            pyst_node = TrackSixTrackLib(
                cudatrackjob, i_last, i_curr + 1,
                context=context, allow_losses=install_apertures)
            one_turn_map.append(pyst_node)

            # sc_node = SpaceChargePIC(length_covered, pypic_algorithm)
            sc_node = AnalyticTransverseGaussianSC(
                length=length_covered,
                sigma_x=sc_sig_x[i_sc],
                sigma_y=sc_sig_y[i_sc],
                sigma_z=self.sig_z,
                wrt_centroid=True,
                update_every=1,
            )
            one_turn_map.append(sc_node)

            i_last = i_curr #+ 1 #+1 is not usually here...

        pyst_node = TrackSixTrackLib(
            cudatrackjob, i_last, n_relevant_elements, context=context)
        one_turn_map.append(pyst_node)

        # n_elements_sixtracklib = len(elements.get_elements())
        # assert pyst_node.i_end == n_elements_sixtracklib, \
        #     'PySixTrack elements do not match SixTrackLib elements'

        pyst_node.is_last_element = True

        return one_turn_map

    def track(self, pyht_beam, one_turn_map, nturns=None):
        if nturns is None:
            nturns = self.nturns

        bunchmon = BunchMonitor(f'bmon_matchedSC_{qx:.2f}_{qy:.2f}',
                                nturns + 1, write_buffer_every=500)
        # partmon = ParticleMonitor(f'pmon_matchedSC_{qx:.2f}_{qy:.2f}',
        #                           stride=self.npart // self.n_stored_particles)

        with GPU(pyht_beam):
            bunchmon.dump(pyht_beam)
            # partmon.dump(pyht_beam)

            for i in range(1, nturns + 1):
                for m in one_turn_map:
                    m.track(pyht_beam)

                bunchmon.dump(pyht_beam)

                # partmon.dump(pyht_beam)

                sys.stdout.write('\rTurn {}/{}'.format(i, nturns))
        return bunchmon.filename, "" #partmon.filename

    def teardown(self, trackjob, filename_error_table, bmon_name, pmon_name,
                 fake=False):
        trackjob.collect_particles()

        store = {}
        filename_error_table = os.path.basename(filename_error_table)

        # with h5py.File(pmon_name + '.h5part', 'r') as fp:
        #     n_steps = len([st for st in fp.keys() if 'Step' in st])
        #     n_stored_particles = len(fp['Step#0']['x'])

        #     rec_inc_x = np.empty((n_steps, self.n_stored_particles),
        #                          dtype=np.float32)
        #     rec_inc_xp = np.empty_like(rec_inc_x)
        #     rec_inc_y = np.empty_like(rec_inc_x)
        #     rec_inc_yp = np.empty_like(rec_inc_x)
        #     rec_inc_z = np.empty_like(rec_inc_x)
        #     rec_inc_dp = np.empty_like(rec_inc_x)

        #     for i in range(n_steps):
        #         rec_inc_x[i, :] = fp['Step#{}'.format(i)]['x']
        #         rec_inc_xp[i, :] = fp['Step#{}'.format(i)]['xp']
        #         rec_inc_y[i, :] = fp['Step#{}'.format(i)]['y']
        #         rec_inc_yp[i, :] = fp['Step#{}'.format(i)]['yp']
        #         rec_inc_z[i, :] = fp['Step#{}'.format(i)]['z']
        #         rec_inc_dp[i, :] = fp['Step#{}'.format(i)]['dp']

        with h5py.File(bmon_name + '.h5', 'r') as fb:
            # rec_mean_x = np.array(fb['Bunch']['mean_x'])
            # rec_mean_y = np.array(fb['Bunch']['mean_y'])
            # rec_epsn_x = np.array(fb['Bunch']['epsn_x'])
            # rec_epsn_y = np.array(fb['Bunch']['epsn_y'])
            rec_sigma_x = np.mean(fb['Bunch']['sigma_x'][-50:])
            rec_sigma_y = np.mean(fb['Bunch']['sigma_y'][-50:])


        # statistics
        # x = trackjob.output.particles[0].x.reshape((self.nturns, self.npart)).T
        store['std_x'] = rec_sigma_x #np.mean(np.std(x, axis=0)[-50:])
        # y = trackjob.output.particles[0].y.reshape((self.nturns, self.npart)).T
        store['std_y'] = rec_sigma_y #np.mean(np.std(y, axis=0)[-50:])

        # losses
        pbuffer = trackjob.particles_buffer.get_object(0)
        if not fake:
            np.save('results/' + filename_error_table + '_alive.npy', pbuffer.state)
            np.save('results/' + filename_error_table + '_lost_at_element.npy',
                    pbuffer.at_element[~pbuffer.state.astype(bool)])
            np.save('results/' + filename_error_table + '_lost_at_turn.npy',
                    pbuffer.at_turn[~pbuffer.state.astype(bool)])

        store['losses'] = np.sum(pbuffer.state)

        # finish job
        if not fake:
            pickle.dump(store, open('results/' + filename_error_table + '_summary.p', 'wb'))

            with open('results/' + filename_error_table + '_done', 'w') as t:
                t.write('')

        self.madx.exit()


def rms_beam_size(beta_optics, epsn, disp_optics, sigma_dp, beta, gamma):
    return np.sqrt(beta_optics * epsn / (beta * gamma) +
                   (disp_optics * sigma_dp)**2)

def provide_pycuda_array(ptr, dtype=np.float64):
    return gp.GPUArray(n_macroparticles, dtype=dtype, gpudata=ptr)

def gpuarray_memcpy(dest, src):
    '''Device memory copy with pycuda from
    src GPUArray to dest GPUArray.
    '''
#     dest[:] = src
#     memcpy_atoa(dest, 0, src, 0, len(src))
    memcpy_dtod_async(dest.gpudata, src.gpudata, src.nbytes)


class TrackSixTrackLib(Element):
    '''General state.'''
    trackjob = None
    pointers = {}
    context = None
    n_elements = 0

    def __init__(self, trackjob, i_start, i_end, context=context, allow_losses=False):
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
                'id': provide_pycuda_array(ptr.contents.particle_id, np.int64),
                'state': provide_pycuda_array(ptr.contents.state, np.int64),
                'at_turn': provide_pycuda_array(ptr.contents.at_turn, np.int64),
                'at_element': provide_pycuda_array(ptr.contents.at_element, np.int64),
                's': provide_pycuda_array(ptr.contents.s, np.float64),
            })
            TrackSixTrackLib.n_elements = len(trackjob.beam_elements_buffer.get_elements())

        self.i_start = i_start
        self.i_end = i_end
        self.is_last_element = (i_end == self.n_elements)

        self.context = context

        self.allow_losses = allow_losses
        if allow_losses:
            self.aperture = SixTrackLibAperture(self)

    memcpy = staticmethod(gpuarray_memcpy)

    def track(self, beam):
        trackjob = TrackSixTrackLib.trackjob

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

        # apply SixTrackLib beam loss to PyHEADTAIL:
        if self.allow_losses:
            self.transfer_losses(beam)

    def transfer_losses(self, beam):
            ### this reorders the particles arrays!
            self.aperture.track(beam)

    def pyht_to_stlib(self, beam):
        self.memcpy(self.pointers['x'], beam.x)
        self.memcpy(self.pointers['px'], beam.xp)
        self.memcpy(self.pointers['y'], beam.y)
        self.memcpy(self.pointers['py'], beam.yp)
        self.memcpy(self.pointers['z'], beam.z)
        self.memcpy(self.pointers['delta'], beam.dp)

        self.memcpy(self.pointers['id'], beam.id)
        self.memcpy(self.pointers['state'], beam.state)
        self.memcpy(self.pointers['at_turn'], beam.at_turn)
        self.memcpy(self.pointers['at_element'], beam.at_element)
        self.memcpy(self.pointers['s'], beam.s)

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

    def stlib_to_pyht(self, beam):
        if self.allow_losses:
            all = np.s_[:beam.macroparticlenumber]
        else:
            all = np.s_[:]
        beam.x = self.pointers['x'][all]
        beam.xp = self.pointers['px'][all]
        beam.y = self.pointers['y'][all]
        beam.yp = self.pointers['py'][all]
        beam.z = self.pointers['z'][all]
        beam.dp = self.pointers['delta'][all]
        beam.id = self.pointers['id'][all]
        beam.state = self.pointers['state'][all]
        beam.at_turn = self.pointers['at_turn'][all]
        beam.at_element = self.pointers['at_element'][all]
        beam.s = self.pointers['s'][all]

def add_STL_attrs_to_PyHT_beam(pyht_beam):
    '''Upgrade PyHEADTAIL.Particles instance pyht_beam
    with all relevant attributes required by SixTrackLib.
    Thus, any reordering of pyht_beam (e.g. due to
    slicing or beam loss) will apply to the new
    attributes from SixTrackLib as well.
    '''
    assert not any(map(
        lambda a: hasattr(pyht_beam, a),
        ['state', 'at_turn', 'at_element', 's']))
    n = pyht_beam.macroparticlenumber

    coords_n_momenta_dict = {
        'state': np.ones(n, dtype=np.int64),
        'at_turn': np.zeros(n, dtype=np.int64),
        'at_element': np.zeros(n, dtype=np.int64),
        's': np.zeros(n, dtype=np.float64),
    }

    pyht_beam.update(coords_n_momenta_dict)
    pyht_beam.id = pyht_beam.id.astype(np.int64)


class SixTrackLibAperture(Aperture):
    '''Removes particles in PyHEADTAIL which have
    been lost in a SixTrackLib aperture (state=0).
    '''
    def __init__(self, tracksixtracklib, *args, **kwargs):
        '''sixtracklib_pointer is a dictionary with GPUArrays
        pointing to the SixTrackLib.Particles attributes.
        These have length total number of macro-particles
        as initially started.
        In PyHEADTAIL the beam.macroparticlenumber will decrease
        with lost particles while in SixTrackLib the attribute arrays
        remain at the same original length and just state gets
        switched to 0 for each lost particle.
        '''
        self.pyht_to_stlib = tracksixtracklib.pyht_to_stlib
        self.stl_p = tracksixtracklib.pointers

    memcpy = staticmethod(gpuarray_memcpy)

    def relocate_lost_particles(self, beam, alive):
        '''Overwriting the Aperture.relocate_lost_particles
        in order to update the SixTrackLib arrays with the fully
        reordered PyHEADTAIL macro-particle arrays before
        they get cut to the decreased length of still
        alive macro-particles.
        '''
        # descending sort to have alive particles (the 1 entries) in the front
        perm = pm.argsort(-alive)

        beam.reorder(perm)

        n_alive = pm.sum(alive)
        # on CPU: (even if pm.device == 'GPU', as pm.sum returns np.ndarray)
        n_alive = np.int32(n_alive)

        ### additional part for SixTrackLib:
        self.pyht_to_stlib(beam)
        self.memcpy(self.stl_p['state'], beam.state)
        self.memcpy(self.stl_p['at_turn'], beam.at_turn)
        self.memcpy(self.stl_p['at_element'], beam.at_element)
        self.memcpy(self.stl_p['s'], beam.s)
        ### also need to limit view on SixTrackLib attributes
        ### in PyHT beam for their next reordering
        beam.state = beam.state[:n_alive]
        beam.at_element = beam.at_element[:n_alive]
        beam.at_turn = beam.at_turn[:n_alive]
        beam.s = beam.s[:n_alive]

        return n_alive

    def tag_lost_particles(self, beam):
        '''Return mask of length beam.macroparticlenumber with
        alive particles being 1 and lost particles being 0.
        '''
        return self.stl_p['state'][:beam.macroparticlenumber]


# # II. Run the simulation in parameter scan

# In[10]:

#for qx in tune_range_qx:
#    for qy in tune_range_qy:


if __name__ == '__main__':
    simulation = Runner()
    simulation.run(qx, qy, fake=False)
