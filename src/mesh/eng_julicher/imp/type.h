struct MeshEngJulicher {
    int max_nm; /* maximum number of meshes */
    int nv, nt, ne;

    /* edge lens, edge angles, vert. areas, vert. curvatures */
    double *lens, *angles, *areas, *curvs;

    MeshEdgLen *len;
    MeshAngle  *angle;
    MeshVertArea *area;

    MeshScatter *scatter;
};
