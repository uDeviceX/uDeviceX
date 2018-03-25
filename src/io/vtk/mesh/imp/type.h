struct Mesh {
    int nv, nt, ne; /* vertices and triangles */
    int *tt;   /* triangles [v0 v1 v2] [v0 v1 v2] .. */
    int *ee;   /* edges [e0 e1] [e0 e1] .. */
};
