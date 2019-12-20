#!/usr/bin/env python

import glob, os, shutil, stat, subprocess, sys
import numpy as np
from os.path import expanduser
HOME = expanduser("~")
CWD  = os.getcwd()

# out of 4
if len(sys.argv) > 1:
    CUDA_DEVICE = int(sys.argv[1])
else:
    CUDA_DEVICE = 0

import socket
hostname = socket.gethostname()
### !!!
hostname = 'bla'

WRKDIRBASE = ('/home/HPC/oeftiger/aoeftiger/sixtracklib_pyht_playground/fodo_sc/')
#('/afs/cern.ch/work/o/oeftiger/private/codes/PyHEADTAIL/'
#              'evaluations/')

if 'lxplus' in hostname:
    print ('\n*** LAUNCHING FOR LXPLUS ***\n')
    QUEUE = '2nw' # 2nw4cores
    EXTRA_ARGS = '-R "select[cpuf > 2.5]" -R "!fullsmt"'
elif 'hpc' in hostname:
    print ('\n*** LAUNCHING FOR CNAF ***\n')
    QUEUE = 'hpc_gpu'
    EXTRA_ARGS = '-R "select [hname!=\'hpc-200-06-03\']" '
    # rusage [ngpus_excl_p=1]" '
    WRKDIRBASE = '/home/HPC/oeftiger/cern/codes/PyHEADTAIL/evaluations/'
else:
    print ('\n*** LAUNCHING LOCALLY VIA TMUX SESSION on GPU #{} ***\n'.format(CUDA_DEVICE))
    sessionname = 'fodo_sc_' + str(CUDA_DEVICE)

EMAIL = 'oeftiger@cern.ch'
PARALLEL = False
JCHUNK = 1200

# ==== Simulation Working Directory
WRKDIR = WRKDIRBASE + '035_Qs-scan_based_on_027/'

# ==== Directory that contains the base cfg file with name filename
SRCDIR = CWD
PREFIX = 'results_' + str(CUDA_DEVICE)

# ==== Simulation parameters
qs = 92./360. * 10**np.linspace(-4, 0, 100)[CUDA_DEVICE::4]

def launch(qs, hostname):

    wrkdir = create_directories(WRKDIR, SRCDIR)

    # Prepare scan

    for i, qsi in enumerate(qs):
                    write_scan_files(
                        wrkdir,
                        i,
                        '{0:.16g}'.format(qsi)
                    )

    if 'lxplus' in hostname:
        bsub_to_lxbatch(wrkdir,
                        jobmax=len(intensity)*len(chroma)*len(dampingrate)*len(dampingphase)*len(i_oct),
                        prefix=PREFIX)
    elif 'hpc' in hostname:
        bsub_to_hpcbatch(wrkdir,
                         jobmax=len(intensity)*len(chroma)*len(dampingrate)*len(dampingphase)*len(i_oct),
                         prefix=PREFIX)
    else:
        # run locally in tmux windows
        subprocess.call(['tmux', 'new', '-d', '-s', sessionname, 'ipython'])
        # trigger first simulation run:
        subprocess.call(['tmux', 'send', '-t', sessionname,
                         '!touch {}/Output/finished0'.format(wrkdir), 'Enter'])

        for qsi in qs:
                subprocess.call(['tmux', 'new-window', '-t', sessionname])
                tmux_window_id = subprocess.check_output(
                    ['tmux', 'display-message', '-p', '#I']).decode('utf8').rstrip()
                # log everything:
                subprocess.call(['tmux', 'pipe-pane', '-t', sessionname,
                                 'cat>{}/Output/tmux.{}.log'.format(
                                    wrkdir, tmux_window_id)])
                subprocess.call(['tmux', 'send', '-t', sessionname,
                                 '# running at Qs {:g}'.format(qsi),
                                 'Enter'])
                subprocess.call(['tmux', 'send', '-t', sessionname,
                                 'cd {}'.format(wrkdir), 'Enter'])
                subprocess.call(['tmux', 'send', '-t', sessionname,
                                 'export CUDA_VISIBLE_DEVICES=' + str(CUDA_DEVICE % 4), 'Enter'])
                subprocess.call(['tmux', 'send', '-t', sessionname,
                                 # wait until the previous run has finished:
                                 'while [ ! -e Output/finished{0} ]; do sleep 1; done; '
                                 # run the simulation:
                                 'python tmppyheadtail.{1}.py && '
                                 # trigger next run after finishing this one:
                                 'touch Output/finished{1}'.format(
                                    int(tmux_window_id) - 1, tmux_window_id),
                                 'Enter'])
        print ('--> Launched tmux session "{:s}" ***\n'.format(sessionname))


def create_directories(wrkdir, srcdir, casdir=None, locdir=None):

    # Catch potential slash at end of path
    if srcdir.split('/')[-1] == '':
        extension = srcdir.split('/')[-2]
    else:
        extension = srcdir.split('/')[-1]

    # Make directories
#    newrkdir = wrkdir + '/' + extension + '_' + PREFIX
    newrkdir = wrkdir + '/' + PREFIX
    if os.path.exists(newrkdir):
        while True:
            ans = input('\nWARNING: Path ' + newrkdir +
                        ' already exists! Overwrite? [yes or no]\n')
            if ans in ('y', 'ye', 'yes'):
                shutil.rmtree(newrkdir)
                break
            if ans in ('n', 'no'):
                print ('\nAborting...')
                exit(0)
            print ('\nPlease answer "yes" or "no"!')

    shutil.copytree(srcdir, newrkdir)
    os.mkdir(newrkdir + '/Data')
    os.mkdir(newrkdir + '/Output')

    return newrkdir


def write_scan_files(wrkdir, it, kwargstr):

    with open(wrkdir + '/tmppyheadtail.' + str(it + 1) + '.py', 'wt') as file:
        file.write('import numpy as np\n')
        file.write('import os, shutil\n')

        outputpath = wrkdir + '/Data/' + str(it + 1) + '/'
        os.makedirs(outputpath)
        file.write('shutil.copy("main.py", "' + outputpath + '")\n')

        file.write('print ("****** Running at ' + kwargstr + '!")\n\n')
        file.write('os.chdir("' + outputpath + '")\n')
        file.write('os.system("python main.py ' + kwargstr + '")\n\n')


def bsub_to_lxbatch(wrkdir, jobmin=1, jobmax=1, libraries=None,
                    prefix='', casdir=None, locdir=None):

    os.chdir(wrkdir)

    with open('myjob.lsf', 'w') as file:
        file.write('#!/bin/bash')
        file.write('\nexport PATH="/afs/cern.ch/user/o/oeftiger/anaconda/bin:$PATH"')
        file.write('\nwhich python')
        file.write('\n\ncd ' + wrkdir)
        file.write('\n\nulimit -c 0')
        file.write('\n\npython tmppyheadtail.$LSB_JOBINDEX.py')
        file.write('\nls -l')

        file.write('\n\necho -e "\\n\\n******** LSF job successfully completed!"')
        # file.write('\n\necho -e "\\n******** Now copying output files..."')
        # file.write('\ncp *.h5 ' + wrkdir + '/Data')

        file.write('\necho -e "HOSTNAME: "')
        file.write('\nhostname')
        file.write('\necho -e "\\n"')
        file.write('\ncat /proc/cpuinfo')
        file.write('\necho -e "\\n*** DEBUG END ****"')

    print ('\n*** Submitting jobs ' + prefix + ' to LSF...')

    for i in range(int(jobmax / JCHUNK) + 1):
        a = i * JCHUNK + 1
        b = (i + 1) * JCHUNK
        if b > jobmax: b = jobmax

        lsfcommand = ['bsub', '-L /bin/bash', '-N ',
                      '-e ' + wrkdir + '/Output/stderror.%J.%I.log ',
                      '-o ' + wrkdir + '/Output/stdout.%J.%I.log',
                      '-J ' + prefix + '[' + str(a) + '-' + str(b) + ']',
                      '-u ' + EMAIL, '-q ' + QUEUE, '< myjob.lsf']
        if EXTRA_ARGS:
            lsfcommand.insert(1, EXTRA_ARGS)
        if PARALLEL:
            lsfcommand.insert(1, '-n 8 -R "span[hosts=1]"')
        lsfcommand = ' '.join(lsfcommand)
        print ('Executing submission with command: ' + lsfcommand)

        with open('launch' + str(i + 1), 'wt') as file:
            file.write("#!/bin/bash\n")
            for lsfc in lsfcommand:
                file.write(lsfc)
        os.chmod("launch" + str(i + 1), 0o0777)
        subprocess.call("./launch" + str(i + 1))


def bsub_to_hpcbatch(wrkdir, jobmin=1, jobmax=1, libraries=None,
                     prefix='', casdir=None, locdir=None):

    os.chdir(wrkdir)

    with open('myjob.lsf', 'w') as file:
        file.write('#!/bin/bash')
#        file.write('\nmodule load compilers/cuda-8.0')
        file.write('\nmodule load compilers/cuda-7.5')
        file.write('\nnvcc --version')
        file.write('\nnvidia-smi')
        file.write('\nexport PATH="/home/HPC/oeftiger/anaconda/bin:$PATH"')
        file.write('\nwhich python')
        file.write('\n\ncd ' + wrkdir)
        file.write('\n\nulimit -c 0')
        file.write('\n\npython tmppyheadtail.$LSB_JOBINDEX.py')
        file.write('\nls -l')

        file.write('\n\necho -e "\\n\\n******** LSF job successfully completed!"')
        # file.write('\n\necho -e "\\n******** Now copying output files..."')
        # file.write('\ncp *.h5 ' + wrkdir + '/Data')

        file.write('\necho -e "HOSTNAME: "')
        file.write('\nhostname')
        file.write('\necho -e "\\n"')
        file.write('\ncat /proc/cpuinfo')
        file.write('\necho -e "\\n*** DEBUG END ****"')

    print ('\n*** Submitting jobs ' + prefix + ' to LSF...')

    for i in range(int(jobmax / JCHUNK) + 1):
        a = i * JCHUNK + 1
        b = (i + 1) * JCHUNK
        if b > jobmax: b = jobmax

        lsfcommand = ['bsub', '-L /bin/bash', '-N ',
                      '-e ' + wrkdir + '/Output/stderror.%J.%I.log ',
                      '-o ' + wrkdir + '/Output/stdout.%J.%I.log',
                      '-J ' + prefix + '[' + str(a) + '-' + str(b) + ']',
                      '-u ' + EMAIL, '-q ' + QUEUE, '< myjob.lsf']
        if EXTRA_ARGS:
            lsfcommand.insert(1, EXTRA_ARGS)
        if PARALLEL:
            lsfcommand.insert(1, '-n 8 -R "span[hosts=1]"')
        lsfcommand = ' '.join(lsfcommand)
        print ('Executing submission with command: ' + lsfcommand)

        with open('launch' + str(i + 1), 'wt') as file:
            file.write("#!/bin/bash\n")
            for lsfc in lsfcommand:
                file.write(lsfc)
        os.chmod("launch" + str(i + 1), 0o0777)
        subprocess.call("./launch" + str(i + 1))


if __name__ == "__main__":
    launch(qs, hostname)
