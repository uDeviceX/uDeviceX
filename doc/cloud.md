# forces::pa

`Pa` is an abstract particle expected by `forces::gen`. Two operation
are supported:

    p2r3(Pa *p, /**/ float *x, float *y, float *z)
	shift(float x, float y, float z, /**/ Pa *p)

# cloud

Cloud is a family of abstract objects.  A cloud has initialization

    Cloud.ini(pp, [...])

and an operation

    Pa = Cloud.cloud_get([...])

Clouds not `pp` should be arguments of functions which call
`forces::gen`.

# example

[src/cloud](../src/cloud)

There are two types of clouds `hforces::` and `lforces::`. `lforces`
is used only for local dpd forces. And `hforces` is used for `hforces`
and `fsi`.

    inline void ini_cloud(Particle *pp, CloudA *c)

To extract a particle
hforces/cloud/get.h

    __device__ void cloud_get(CloudA c, int i, /**/ forces::Pa *p)

To set/get position from `forces::Pa`
forces/use.h

     inline __device__ void p2r3(Pa *p, /**/ float *x, float *y, float *z)
	 inline __device__ void shift(float x, float y, float z, /**/ Pa *p)

Once two particles are extracted they are passed to generic force

    inline __device__ void gen(Pa A, Pa B, float rnd, /**/ float *fx,
    float *fy, float *fz)

