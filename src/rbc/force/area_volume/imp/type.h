struct AreaVolume {
    int nt, nv; /* number of triangles and vertices in one cell */
    int4 *tri; /* triangles on device */

    float *av; /* output: [area0 volume0 area1 volume1, ..]
};
