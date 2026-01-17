#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=24:mem=100gb:scratch_local=50gb
#PBS -N Thrombosis_analysis



# define a DATADIR variable: directory where the input files are taken from and where the output will be copied to
DATADIR=/storage/brno12-cerit/home/user123/test_directory # substitute username and path to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of the node it is run on, and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails, and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add mambaforge

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp $DATADIR/code  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

cd $SCRATCHDIR

mamba activate --prefix $SCRATCHDIR/env

python main.py --load --sim -v --log log.log -i

mambda deactivate

cp -R results/ $DATADIR/results || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }
cp -R cache/ $DATADIR/cache || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 5; }
cp log.txt $DATADIR/ || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 6; }
cp log.log $DATADIR/ || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 7; }

clean_scratch
