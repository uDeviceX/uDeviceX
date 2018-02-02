struct Rnd0 { };

static __device__ void edg_rnd(Shape, int, real*, int, Rnd0*) { }
static __device__ real3 frnd(real, RbcParams_v, real3, real3, const Rnd0) { return make_real3(0, 0, 0); }
