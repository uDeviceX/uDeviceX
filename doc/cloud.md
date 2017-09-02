# intro
`Pa` is an abstract particle expected by `forces::gen`

Cloud is a family of abstract objects.  A cloud has initialization

    Cloud.ini(pp, [...])

and an operation

    Pa = Cloud.cloud_get([...])

Clouds not `pp` should be arguments of functions which call
`forces::gen`.

# example for src/hforces

`src/hforces` treats particles assymetricly: `A` is "a main particle"
and `B` is a particle from neighorhood of `A`. There are two
corresponding clouds `CloudA` and `CloudB`.

Initialization of a cloud: called in dpdr/int.cu
hforces/cloud/int.h

    inline void ini_cloudA(Particle *pp, CloudA *c)

To extract a particle
hforces/cloud/get.h

    __device__ void cloudA_get(CloudA c, int i, /**/ forces::Pa *p)

To set/get position from `forces::Pa`
forces/use.h

     inline __device__ void p2r3(Pa *p, /**/ float *x, float *y, float *z)
	 inline __device__ void shift(float x, float y, float z, /**/ Pa *p)

Once two particles are extracted they are passed to generic force

    inline __device__ void gen(Pa A, Pa B, float rnd, /**/ float *fx, float *fy, float *fz)

# plan

`forces::gen` takes two `Pa` and compute force.

    ./dpd/dev/dpd.h:12:    forces::gen(a, b, rnd, &fx, &fy, &fz);
    ./fsi/dev/pair.h:7:    forces::gen(a, b, rnd, /**/ fx, fy, fz);
    ./k/cnt/halo.h:96:     forces::gen(a, b, rnd, &fx, &fy, &fz);
    ./k/cnt/bulk.h:87:     forces::gen(a, b, rnd, &fx, &fy, &fz);
    ./wall/dev.h:103:      forces::gen(a, b, rnd, /**/ &fx, &fy, &fz);
