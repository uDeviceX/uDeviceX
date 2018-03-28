struct MeshEngJulicher {
    int max_nm; /* maximum number of meshes */
    int nv, nt, ne;

    /* edge lens, edge angles, vert. areas */
    double *lens, *angles, *areas;
};
