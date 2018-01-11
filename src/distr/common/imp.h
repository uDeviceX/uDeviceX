void dcommon_pack_pp_packets(int nc, int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf);
void dcommon_shift_one_frag(int n, const int fid, /**/ Particle *pp);
void dcommon_shift_halo(int nhalo, const Sarray<int, 27> starts, /**/ Particle *pp);
