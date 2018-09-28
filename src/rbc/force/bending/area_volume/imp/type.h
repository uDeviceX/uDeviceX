struct AreaVolume {
    int nt, nv; /* number of triangles and vertices in one cell */
    int max_cell; /* max cell number */
    int4 *tri; /* triangles on device */
    float *av; /* output: [area0 volume0 area1 volume1, ..] */
    float *av_hst;
    int nc;       /* remember the last number of cell */
    int Computed; /* was area computed? */
};
