struct VTKConf {
    Mesh *mesh;
    KeyList *tri;
};

struct VTK {
    int maxn;
    Mesh *mesh;
    double *rr; /* positions */
};
