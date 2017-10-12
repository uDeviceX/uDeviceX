/* set colors of particles according to the RBCs */

void gen_colors() {
    int nm, nv, nmhalo;
    nm = r::q.nc;
    nv = r::q.nv;

    using namespace mc;
    
    build_map(nm, nv, r::q.pp, /**/ &e.p);
    pack(nv, r::q.pp, /**/ &e.p);
    download(&e.p);

    post_send(&e.p, &e.c);
    post_recv(&e.c, &e.u);

    if (nm * nv) CC(d::MemcpyAsync(pp, r::q.pp, nm * nv * sizeof(Particle), D2D));
    
    wait_send(&e.c);
    wait_recv(&e.c, &e.u);

    unpack(nv, &e.u, /**/ &nmhalo, pp + nm * nv);
    nm += nmhalo;

    /* tmp texture object; TODO: make a ticket? */
    Texo<float2> texvert;
    TE(&texvert, (float2*) pp, 3 * nm * nv);
    
    collision::get_colors(o::q.pp, o::q.n, texvert, r::tt.textri, r::q.nt, nv, nm, /**/ o::q.cc);

    texvert.destroy();
}
