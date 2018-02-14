void eflu_get_local_frags(const EFluPack *p, /**/ LFrag26 *lfrags) {
    PaArray parraya;    
    
    for (int i = 0; i < 26; ++i) {
        parray_push_pp((Particle*) p->dpp.data[i], &parraya);
        if (multi_solvent)
            parray_push_cc((int*) p->dcc.data[i], &parraya);

        lfrags->d[i] = {
            .parray  = parraya,            
            .ii = p->bii.d[i],
            .n  = p->hpp.counts[i]
        };
    }
}

void eflu_get_remote_frags(const EFluUnpack *u, /**/ RFrag26 *rfrags) {
    int i, dx, dy, dz, xcells, ycells, zcells;
    int3 L;
    PaArray parrayb;
    L = u->L;
    
    for (i = 0; i < 26; ++i) {
        dx = fraghst::i2dx(i);
        dy = fraghst::i2dy(i);
        dz = fraghst::i2dz(i);

        xcells = dx == 0 ? L.x : 1;
        ycells = dy == 0 ? L.y : 1;
        zcells = dz == 0 ? L.z : 1;
        parray_push_pp((Particle*) u->dpp.data[i], &parrayb);
        if (multi_solvent)
            parray_push_cc((int*) u->dcc.data[i], &parrayb);
            
        rfrags->d[i] = {
            .parray = parrayb,
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
