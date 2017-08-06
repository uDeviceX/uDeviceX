static void grid(FILE * xmf,
                 const char * const h5path, const char * const *channelnames, int nchannels) {
    enum {X, Y, Z};
    int i;
    int *d, G[3];
    d = m::dims;
    G[X] = XS*d[X]; G[Y] = YS*d[Y]; G[Z] = ZS*d[Z];

    fprintf(xmf, "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n");
    fprintf(xmf, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n", 1 + G[Z], 1 + G[Y], 1 + G[X]);
    fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(xmf, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(xmf, "        %e %e %e\n", 0.0, 0.0, 0.0);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(xmf, "        %e %e %e\n", 1.0, 1.0, 1.0);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Geometry>\n");
    for(i = 0; i < nchannels; ++i) {
        fprintf(xmf, "     <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", channelnames[i]);
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", G[Z], G[Y], G[X]);
        fprintf(xmf, "        %s:/%s\n", h5path, channelnames[i]);
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");
    }
    fprintf(xmf, "   </Grid>\n");
}
