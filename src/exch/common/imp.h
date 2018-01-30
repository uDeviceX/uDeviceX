struct int3;

struct PackHelper {
    int *starts;
    int *offsets;
    int *indices[NFRAGS];
};

void ecommon_pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf);
void ecommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp);
