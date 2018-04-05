struct VTKConf {
    Mesh *mesh;
    KeyList *tri, *vert;
};

enum { MAGIC = 1234 };
enum { UNSET = - 1 };
enum { N_MAX = 100 };
struct VTK {
    int nm; /* current number of meshes */
    Mesh *mesh;
    KeyList *tri, *vert;

    int nbuf; /* maximum bufer size */
    double *dbuf;
    int *ibuf;

    double *TRI[N_MAX];
    double *VERT[N_MAX];
    
    int rr_set;
    char path[FILENAME_MAX];

    int stamp; /* MAGIC */
};

struct Out {
    WriteFile *file;
    MPI_Comm comm;
};
