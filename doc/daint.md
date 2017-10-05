# daint

related daint commands

## interactive session

4 gpu nodes for 1 hour:
```
salloc --constraint=gpu --time=01:00:00 -N 4
```

## easy build for Octave

```
module load EasyBuild-custom/cscs
eb -S Octave
export EASYBUILD_PREFIX=/tmp/$USER
eb /apps/common/UES/jenkins/production/easybuild/easyconfigs/o/Octave/Octave-4.2.0-CrayGNU-2016.11.eb --robot
```
