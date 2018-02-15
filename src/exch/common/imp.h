struct int3;

// tag::struct[]
struct PackHelper {
    int *starts;
    int *offsets;
    int *indices[NFRAGS];
};
// end::struct[]

// tag::int[]
void ecommon_pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf); // <1>
void ecommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
// end::int[]
