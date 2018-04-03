struct MeshWrite {
    int4 *tt;
    int nv, nt;
    char directory[FILENAME_MAX];
    int shift_type;
};
