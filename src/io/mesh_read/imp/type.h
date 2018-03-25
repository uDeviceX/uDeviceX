struct MeshRead {
    int nv, nt, nd; /* vertices, triangles, dihidrals */
    float *rr; /* [x y z] [x y z] */    
    int4  *tt; /* triangles: [v0 v1 v2 dummy] */
    int4  *dd; /* dihidrals: [v0 (v1 v2) v3]  */
};
