struct PackHelper {
    int *starts;
    int *offsets;
    int *indices[comm::NFRAGS];
};

void ecommon_pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf);
void ecommon_shift_one_frag(int n, const int fid, /**/ Particle *pp);
