# daint

related daint commands

## interactive session

4 gpu nodes for 1 hour:
```
salloc --constraint=gpu --time=01:00:00 -N 4
```

debug queue:
```
salloc --constraint=gpu -p debug --time=00:30:00 -N 1
```

## easy build for Octave

(temporary) solution to load Octave: build it from easy-build:
```
module load EasyBuild-custom/cscs
eb -S Octave
export EASYBUILD_PREFIX=/tmp/$USER
eb /apps/common/UES/jenkins/production/easybuild/easyconfigs/o/Octave/Octave-4.2.0-CrayGNU-2016.11.eb --robot
```
```
export MODULEPATH=$EASYBUILD_PREFIX/modules/all:$MODULEPATH
module load Octave
```
