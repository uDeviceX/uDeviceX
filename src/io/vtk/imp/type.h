#pragma once

struct VTKConf {
    Mesh *mesh;
    KeyList *tri, *vert;
};

enum { VTK_MAGIC = 1234 };
enum { VTK_UNSET = - 1 };
enum { VTK_N_MAX = 100 };
struct VTK {
    int nm; /* current number of meshes */
    Mesh *mesh;
    KeyList *tri, *vert;

    int nbuf; /* maximum bufer size */
    double *dbuf;
    int *ibuf;

    double *TRI[VTK_N_MAX];
    double *VERT[VTK_N_MAX];
    
    int rr_set;
    char path[FILENAME_MAX];

    int stamp; /* MAGIC */
};

struct Out {
    WriteFile *file;
    MPI_Comm comm;
};
