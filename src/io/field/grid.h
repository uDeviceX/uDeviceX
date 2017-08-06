void H5FieldDump::grid(FILE * xmf,
                             const char * const h5path, const char * const *channelnames, int nchannels) {
    enum {X, Y, Z};
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

    const float h = 1;
    fprintf(xmf, "        %e %e %e\n", h, h, h);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Geometry>\n");

    for(int ichannel = 0; ichannel < nchannels; ++ichannel) {
        fprintf(xmf, "     <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", channelnames[ichannel]);
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", G[Z], G[Y], G[X]);
        fprintf(xmf, "        %s:/%s\n", h5path, channelnames[ichannel]);
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");
    }
    fprintf(xmf, "   </Grid>\n");
}
