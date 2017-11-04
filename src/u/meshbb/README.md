# intro

meshbb client

# install

```
u.conf . u/meshbb conf/default.h <<!
dt=1
run
!

u.make -j
```

# use

    echo 0.1 0.1 0.1   0.2 0 0  | ./udx data/cells/side.off
	echo  0.999 0.2 0.1   1 0 0  | ./udx data/cells/side.off
