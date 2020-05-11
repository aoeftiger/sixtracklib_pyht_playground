#!/usr/bin/env python
# coding: utf-8

import sys
assert len(sys.argv) == 1 + 4, (
    'Require 4 arguments: sixtracklib_device '
    '(e.g. opencl:1.0 or cpu), qx, qy, e_seed')

gpu_device_id, qx, qy, e_seed = sys.argv[1:]
qx = float(qx)
qy = float(qy)
e_seed = int(e_seed)

# In[1]:


import numpy as np

import sys, os
import pickle

import time


# In[2]:


from scipy.constants import e, m_p, c

from scipy.constants import physical_constants


# In[3]:


from cpymad.madx import Madx

sys.path = ["/home/HPC/oeftiger/aoeftiger/sixtracklib_dev/python/"] + sys.path

import sixtracklib as pyst
import pysixtrack
import pysixtrack.be_beamfields.tools as bt

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


#tune_range_qx = np.arange(18.55, 18.95 + 0.01, 0.01)
#tune_range_qy = tune_range_qx
#e_seed = 3


with_errors = True #False
force_zero = True # centroid forced to zero orbit offset!
with_SC = True
with_warm_quads = False#True

#err_file = 'Coll+Errors+BeamDistr.madx'
#err_file = 'newColl+Errors+BeamDistr_onlyk1l_relative.madx'
err_file = 'newColl+Errors+BeamDistr_onlyk1l_absolute.madx'

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

    sc_mode = 'Bunched' # 'Coasting'
    intensity = 0.625e11
    n_scnodes = 500

    def __init__(self, nturns=20000, npart=1000, e_seed=e_seed):
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

        self.sig_z = self.sig_z * 1.22
        self.sig_dp = self.sig_dp * 1.22

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
               'Qy = {:.2f} and e_seed = {:d} ===\n\n\n'.format(qx, qy, self.e_seed))

        ### SETUP
        twiss = self.setup_madx(self.madx, filename_error_table, with_errors, with_warm_quads)
        pysixtrack_elements = self.setup_pysixtrack(
            self.madx, filename_error_table, with_errors, with_SC, with_warm_quads)
        if with_SC:
            closed_orbit = self.correct_closed_orbit(pysixtrack_elements, force_zero=force_zero)
        else:
            closed_orbit = None
        pyht_beam = self.setup_pyheadtail_particles(twiss, closed_orbit)
        trackjob = self.setup_sixtracklib(pysixtrack_elements, pyht_beam)

        print ('\n\n\n' + '+'*26 + '\n*** ready for tracking ***\n' +
               '+'*26 + '\n')
        print ('\n\n\n=== Running at Qx = {:.2f} and '
               'Qy = {:.2f} and e_seed = {:d} ===\n\n\n'.format(qx, qy, self.e_seed))

        if fake:
            trackjob.track_until(0)
        else:
            trackjob.track_until(self.nturns)
        trackjob.collect()

        self.teardown(trackjob, filename_error_table, fake=fake)
        return trackjob


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
        distribution_z_cut = generators.cut_distribution(distribution_z_uncut, is_accepted)

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

        ### space charge stuff
        if with_SC:
            sc_lengths = self.add_madx_sc_markers(madx, sis100, seqname)

        ### PySixTrack, lattice transfer and preparation!

        pysixtrack_elements = pysixtrack.Line.from_madx_sequence(
            sis100, exact_drift=True, install_apertures=True
        )

        pysixtrack_elements.remove_zero_length_drifts(inplace=True)
        pysixtrack_elements.merge_consecutive_drifts(inplace=True)

        # add alignment and multipole errors

        if with_errors:
            pysixtrack_elements.apply_madx_errors(error_table=errors)

        if with_SC:
            self.add_pysixtrack_sc_nodes(
                madx, pysixtrack_elements, sc_lengths, seqname)

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
            madx, seqname, sc_names, sc_locations, mode=self.sc_mode)

        return sc_lengths

    def add_pysixtrack_sc_nodes(
            self, madx, pysixtrack_elements, sc_lengths, seqname):
        pst_elements = pysixtrack_elements
        mad_sc_names, sc_twdata = bt.get_spacecharge_names_twdata(
            madx, seqname, mode=self.sc_mode
        )

        # Check consistency
        if self.sc_mode == "Bunched":
            sc_elements, sc_names = pst_elements.get_elements_of_type(
                pysixtrack.elements.SpaceChargeBunched
            )
        elif self.sc_mode == "Coasting":
            sc_elements, sc_names = pst_elements.get_elements_of_type(
                pysixtrack.elements.SpaceChargeCoasting
            )
        else:
            raise ValueError("SC mode not understood")

        bt.check_spacecharge_consistency(
            sc_elements, sc_names, sc_lengths, mad_sc_names
        )

        # Setup spacecharge in the line
        if self.sc_mode == "Bunched":
            bt.setup_spacecharge_bunched_in_line(
                sc_elements,
                sc_lengths,
                sc_twdata,
                self.beta * self.gamma,
                self.intensity,
                self.sig_z/1.22,
                self.sig_dp /1.22,
                35e-6/4 * self.beta * self.gamma,
                15e-6/4 * self.beta * self.gamma,
            )

        elif self.sc_mode == "Coasting":
            raise NotImplementedError(
                'have to supply line_density')
            bt.setup_spacecharge_coasting_in_line(
                sc_elements,
                sc_lengths,
                sc_twdata,
                self.beta * self.gamma,
                line_density,
                self.sig_dp,
                self.epsn_x,
                self.epsn_y,
            )

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
        trackjob = pyst.TrackJob(elements, particles,
                                 device=gpu_device_id)

        return trackjob

    def teardown(self, trackjob, filename_error_table, fake=False):
        store = {}
        filename_error_table = os.path.basename(filename_error_table)

        # statistics
        x = trackjob.output.particles[0].x.reshape((self.nturns, self.npart)).T
        store['std_x'] = np.mean(np.std(x, axis=0)[-50:])
        y = trackjob.output.particles[0].y.reshape((self.nturns, self.npart)).T
        store['std_y'] = np.mean(np.std(y, axis=0)[-50:])

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


# # II. Run the simulation in parameter scan

# In[10]:

#for qx in tune_range_qx:
#    for qy in tune_range_qy:


if __name__ == '__main__':
    simulation = Runner()
    simulation.run(qx, qy, fake=False)
