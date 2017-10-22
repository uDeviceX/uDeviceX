static void grid(FILE * f, const char * const path, const char * const *names, int n) {
    enum {X, Y, Z};
    int i;
    int *d, G[3]; /* domain size */
    d = m::dims;
    G[X] = XS*d[X]; G[Y] = YS*d[Y]; G[Z] = ZS*d[Z];

    fprintf(f, "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n");
    fprintf(f, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n", 1 + G[Z], 1 + G[Y], 1 + G[X]);
    fprintf(f, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(f, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(f, "        %e %e %e\n", 0.0, 0.0, 0.0);
    fprintf(f, "       </DataItem>\n");
    fprintf(f, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(f, "        %e %e %e\n", 1.0, 1.0, 1.0);
    fprintf(f, "       </DataItem>\n");
    fprintf(f, "     </Geometry>\n");
    for(i = 0; i < n; ++i) {
        fprintf(f, "     <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", names[i]);
        fprintf(f, "       <DataItem Dimensions=\"%d %d %d 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", G[Z], G[Y], G[X]);
        fprintf(f, "        %s:/%s\n", path, names[i]);
        fprintf(f, "       </DataItem>\n");
        fprintf(f, "     </Attribute>\n");
    }
    fprintf(f, "   </Grid>\n");
}
