struct Particle;
struct int3;

// tag::int[]
void dcommon_pack_pp_packets(int nc, int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf); // <1>
void dcommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
void dcommon_shift_halo(int3 L, int nhalo, const Sarray<int, 27> starts, /**/ Particle *pp); // <3>
// end::int[]

