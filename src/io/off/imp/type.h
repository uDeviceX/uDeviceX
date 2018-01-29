struct OffRead {
    int nv, nt; /* vertices and triangles */
    int4  *tt; /* triangles: [v0 v1 v2 dummy] */
    float *rr; /* [x y z] [x y z] */
};
