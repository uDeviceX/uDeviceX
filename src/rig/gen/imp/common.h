enum {
    IN  = -2,
    OUT = -1
};

static int get_root(MPI_Comm comm, bool hasid0) {
    int rank, root0, root;
    MC(m::Comm_rank(comm, &rank));
    root0 = hasid0 ? rank : 0;
    root = 0;
    MC(m::Allreduce(&root0, &root, 1, MPI_INT, MPI_SUM, comm));
    if (hasid0 && root != rank) ERR("More than one rank has id 0");
    return root;
}

static void exchange_mesh(int maxm, int3 L, MPI_Comm cart, int nv, /*io*/ int *nm, Particle *pp, /**/ int *cc) {
    EMeshPack *pack;
    EMeshComm *comm;
    EMeshUnpack *unpack;
    int nm0, nmhalo;
    nm0 = *nm;

    UC(emesh_pack_ini(L, nv, maxm, &pack));
    UC(emesh_comm_ini(cart, /**/ &comm));
    UC(emesh_unpack_ini(L, nv, maxm, &unpack));

    UC(emesh_build_map(nm0, nv, pp, /**/ pack));
    UC(emesh_pack(nv, pp, /**/ pack));
    UC(emesh_download(pack));

    UC(emesh_post_recv(comm, unpack));
    UC(emesh_post_send(pack, comm));
    UC(emesh_wait_recv(comm, unpack));
    UC(emesh_wait_send(comm));

    UC(emesh_unpack(nv, unpack, /**/ &nmhalo, pp + nm0 * nv));
    if (cc) UC(emesh_get_num_frag_mesh(unpack, /**/ cc));
    *nm += nmhalo;
    
    UC(emesh_pack_fin(pack));
    UC(emesh_comm_fin(comm));
    UC(emesh_unpack_fin(unpack));
}

static void compute_labels(int pdir, int n, const Particle *pp, int nt, int nv, int nm, const int4 *tt, const Particle *pp_mesh, int in, int out, /**/ int *ll) {
    float3 *lo, *hi;
    Triangles tri;

    if (nm) Dalloc(&lo, nm);
    if (nm) Dalloc(&hi, nm);

    tri.nt = nt;
    tri.tt = (int4*) tt;
    
    if (nm) UC(minmax(pp_mesh, nv, nm, /**/ lo, hi));
    UC(collision_label(pdir, n, pp, &tri, nv, nm, pp_mesh, lo, hi, IN, OUT, /**/ ll));    

    if (nm) Dfree(lo);
    if (nm) Dfree(hi);
}
