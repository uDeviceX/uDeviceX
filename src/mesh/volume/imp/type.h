struct MeshVolume {
    double *rr; /* workspace for coordinates shifted to a mesh center
                   of mass */
    int4 *tt;
    int nv, nt;
};
