struct Mesh {
    int nv, nt;
    int4 *tt;
};

struct VTKConf {
    Mesh *mesh;
    KeyList *tri;
};

struct VTK {
    int maxn;
    double *R;
};
