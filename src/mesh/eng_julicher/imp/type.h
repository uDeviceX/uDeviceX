struct MeshEngJulicher {
    int max_nm; /* maximum number of meshes */
    int nv, ne;

    /* edge lens, edge angles, vert. areas, vert. curvatures */
    double *lens, *angles, *areas, *curvs_edg, *curvs_vert;

    MeshEdgLen *len;
    MeshAngle  *angle;
    MeshVertArea *area;

    MeshScatter *scatter;
};
