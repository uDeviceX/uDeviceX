struct Rnd0 { };

static __device__ void edg_rnd(float*, int, Rnd0*)  { }
static __device__ float3 frnd(float3, float3, const Rnd0 rnd) { return make_float3(0, 0, 0); }
