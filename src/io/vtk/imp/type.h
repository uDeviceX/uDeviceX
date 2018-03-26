struct VTKConf {
    Mesh *mesh;
    KeyList *tri;
};

enum { UNSET = - 1 };
struct VTK {
    int nm; /* current number of meshes */
    Mesh *mesh;

    int nbuf; /* maximum bufer size */
    double *dbuf;
    int *ibuf;
    
    int rr_set;
    char path[FILENAME_MAX];
};

struct Out {
    WriteFile *file;
    MPI_Comm comm;
};
