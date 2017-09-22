/* set colors of particles according to the RBCs */
void gen_colors() {

    if (mc::tc.first) mcomm::post_recv(/**/ &mc::tc, &mc::tr);
    mcomm::extents(r::q.pp, r::q.nv, r::q.nc, /**/ &mc::ts);
    int nbulk = mcomm::map(r::q.nc, /**/ &mc::tm, &mc::ts);
    mcomm::pack(r::q.pp, r::q.nv, &mc::tm, /**/ &mc::ts);
    mcomm::post_send(r::q.nv, &mc::ts, /**/ &mc::tc);
    mcomm::wait_recv(/**/ &mc::tc);
    int nm = mcomm::unpack(r::q.nv, nbulk, /**/ &mc::tr);
    mcomm::post_recv(/**/ &mc::tc, &mc::tr);
    dSync();

    /* tmp texture object; TODO: make a ticket? */
    Texo<float2> texvert;
    TE(&texvert, (float2*)mc::tr.pp, nm * 3 * r::q.nv);

    collision::get_colors(o::q.pp, o::q.n, texvert, r::tt.textri, r::q.nt, r::q.nv, nm, /**/ o::qc.ii);

    texvert.destroy();
    mc::tc.first = false;
}

void gen_colors_new() {
    int nm, nv, nmhalo;
    nm = r::q.nc;
    nv = r::q.nv;

    using namespace mc;
    
    build_map(nm, nv, r::q.pp, /**/ &e.p);
    pack(nv, r::q.pp, /**/ &e.p);
    download(&e.p);

    post_send(&e.p, &e.c);
    post_recv(&e.c, &e.u);

    CC(d::MemcpyAsync(pp, r::q.pp, nm * nv * sizeof(Particle), D2D));
    
    wait_send(&e.c);
    wait_recv(&e.c, &e.u);

    unpack(nv, &e.u, /**/ &nmhalo, pp + nm * nv);
    dSync();
    nm += nmhalo;

    /* tmp texture object; TODO: make a ticket? */
    Texo<float2> texvert;
    TE(&texvert, (float2*) pp, 3 * nm * nv);

    collision::get_colors(o::q.pp, o::q.n, texvert, r::tt.textri, r::q.nt, nv, nm, /**/ o::qc.ii);

    texvert.destroy();
}
