struct VTKConf {
    Mesh *mesh;
    KeyList *tri;
};

struct VTK {
    int maxn;
    Mesh *mesh;
    double *rr; /* positions */
    char path[FILENAME_MAX];
};

struct Out {
    WriteFile *f;
    MPI_Comm comm;
};

