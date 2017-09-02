# "local" Cloud

`Pa`, `Index` are abstract particle and index.

Cloud is a family of abstract object with API.

    Pa = pa(Cloud, Index)
    r  = r(Cloud, Index)

The `__global__` function recives one or two Cloud objects. For
example,

    fun(CloudA, CloudB, [...])

The function uses only API of clouds.

# plan

`forces::gen` takes two `Pa` and compute force.

    ./dpd/dev/dpd.h:12:    forces::gen(a, b, rnd, &fx, &fy, &fz);
    ./fsi/dev/pair.h:7:    forces::gen(a, b, rnd, /**/ fx, fy, fz);
    ./hforces/dev.h:35:    forces::gen(a, b, rnd, /**/ fx, fy, fz);
    ./k/cnt/halo.h:96:     forces::gen(a, b, rnd, &fx, &fy, &fz);
    ./k/cnt/bulk.h:87:     forces::gen(a, b, rnd, &fx, &fy, &fz);
    ./wall/dev.h:103:      forces::gen(a, b, rnd, /**/ &fx, &fy, &fz);

# example for src/hforces

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
