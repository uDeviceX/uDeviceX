enum {
    IN,
    OUT
};

static void exchange_mesh(int maxm, int3 L, MPI_Comm cart, int nv, /*io*/ int *n, Particle *pp) {
    EMeshPack *pack;
    EMeshComm *comm;
    EMeshUnpack *unpack;
    int nm, nmhalo;
    nm = *n / nv;

    UC(emesh_pack_ini(L, nv, maxm, &pack));
    UC(emesh_comm_ini(cart, /**/ &comm));
    UC(emesh_unpack_ini(L, nv, maxm, &unpack));

    UC(emesh_build_map(nm, nv, pp, /**/ pack));
    UC(emesh_pack(nv, pp, /**/ pack));
    UC(emesh_download(pack));

    UC(emesh_post_recv(comm, unpack));
    UC(emesh_post_send(pack, comm));
    UC(emesh_wait_recv(comm, unpack));
    UC(emesh_wait_send(comm));

    UC(emesh_unpack(nv, unpack, /**/ &nmhalo, pp + nm * nv));

    *n += nm * nv;
    
    UC(emesh_pack_fin(pack));
    UC(emesh_comm_fin(comm));
    UC(emesh_unpack_fin(unpack));
}

static void compute_labels(int pdir, int n, const Particle *pp, int nt, int nv, int nm, const int4 *tt, const Particle *pp_mesh, /**/ int *ll) {
    float3 *lo, *hi;
    Triangles tri;
    Dalloc(&lo, nm);
    Dalloc(&hi, nm);

    tri.nt = nt;
    tri.tt = (int4*) tt;
    
    UC(minmax(pp_mesh, nv, nm, /**/ lo, hi));
    UC(collision_label(pdir, n, pp, &tri, nv, nm, pp_mesh, lo, hi, IN, OUT, /**/ ll));    
    Dfree(lo);
    Dfree(hi);
}
