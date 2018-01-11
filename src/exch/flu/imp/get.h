void eflu_get_local_frags(const Pack *p, /**/ LFrag26 *lfrags) {
    Cloud clouda = {0, 0};    
    
    for (int i = 0; i < 26; ++i) {
        ini_cloud((Particle*) p->dpp.data[i], &clouda);
        if (multi_solvent)
            ini_cloud_color((int*) p->dcc.data[i], &clouda);

        lfrags->d[i] = {
            .c  = clouda,            
            .ii = p->bii.d[i],
            .n  = p->hpp.counts[i]
        };
    }
}

void eflu_get_remote_frags(const Unpack *u, /**/ RFrag26 *rfrags) {
    enum {X, Y, Z};
    int i, dx, dy, dz, xcells, ycells, zcells;
    Cloud cloudb = {0, 0};

    for (i = 0; i < 26; ++i) {
        dx = frag_i2d(i,X);
        dy = frag_i2d(i,Y);
        dz = frag_i2d(i,Z);

        xcells = dx == 0 ? XS : 1;
        ycells = dy == 0 ? YS : 1;
        zcells = dz == 0 ? ZS : 1;
        ini_cloud((Particle*) u->dpp.data[i], &cloudb);
        if (multi_solvent)
            ini_cloud_color((int*) u->dcc.data[i], &cloudb);
            
        rfrags->d[i] = {
            .c     = cloudb,
            .start = (int*) u->dfss.data[i],
            .dx = dx,
            .dy = dy,
            .dz = dz,
            .xcells = xcells,
            .ycells = ycells,
            .zcells = zcells,
            .type = abs(dx) + abs(dy) + abs(dz)
        };
    }
}
