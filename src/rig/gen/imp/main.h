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
    
    UC(emesh_pack_fin(pack));
    UC(emesh_comm_fin(comm));
    UC(emesh_unpack_fin(unpack));
}

/* select only mesh 0 and gather particles */ 
static void label_template(int3 L, MPI_Comm cart, int nv, int nm, const Particle *pp_mesh, int n_flu, const Particle *pp_flu, /**/ int *label) {
    int maxm, n;
    Particle *pp;

    maxm = NFRAGS;
    Dalloc(&pp, nv * maxm);
    
    UC(exchange_mesh(maxm, L, cart, nv, /* io */ &n, pp));

    Dfree(pp);
}

void rig_gen_from_solvent(const Coords *coords, MPI_Comm cart, RigGenInfo rgi, /* io */ FluInfo fi, /* o */ RigInfo ri) {
    
}
