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
    int rr_set;
    char path[FILENAME_MAX];
};

struct Out {
    WriteFile *file;
    MPI_Comm comm;
};
