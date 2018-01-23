struct AreaVolume {
    int nt, nv; /* number of triangles and vertices in one cell */
    int4 *tri; /* triangles on device */
};
