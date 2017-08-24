struct HaloMap {
    int **iidx;
    int *sizes;

    int *iidx_[27]; /* needed only to free `iidx' */
};

void halo(const Particle *pp, int n, /**/ int **iidx, int *sizes);
void alloc_halo_map(HaloMap& h);
void free_halo_map(HaloMap& h);
