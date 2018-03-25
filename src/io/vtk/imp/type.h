struct VTKConf {
    Mesh *mesh;
    KeyList *tri;
};

enum { UNSET = - 1 };
struct VTK {
    int maxn;
    int nm;
    Mesh *mesh;
    double *rr; /* positions */
    char path[FILENAME_MAX];
};

struct Out {
    WriteFile *f;
    MPI_Comm comm;
};

