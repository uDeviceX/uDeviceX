static void gather() {
    gather_pp(&o::td, /**/ &o::q, &o::tu, &o::tz);
    if (global_ids)    gather_ii(o::q.n, &o::tu, &o::tui, /**/ &o::qi);
    if (multi_solvent) gather_ii(o::q.n, &o::tu, &o::tuc, /**/ &o::qc);
}

static void pack() {
    pack_pp(&o::q, /**/ &o::td);
    if (global_ids)    pack_ii(o::q.n, &o::qi, &o::td, /**/ &o::ti);
    if (multi_solvent) pack_ii(o::q.n, &o::qc, &o::td, /**/ &o::tc);
}

static void unpack() {
    unpack_pp(&o::td, /**/ &o::q, &o::tu, /*w*/ &o::w);
    if (global_ids)    unpack_ii(&o::td, &o::ti, /**/ &o::tui);
    if (multi_solvent) unpack_ii(&o::td, &o::tc, /**/ &o::tuc);
}

void odstr() {
    using namespace odstr;
    post_recv_pp(/**/ &o::td);
    if (global_ids)    post_recv_ii(&o::td, /**/ &o::ti);
    if (multi_solvent) post_recv_ii(&o::td, /**/ &o::tc);

    pack();

    send_pp(/**/ &o::td);
    if (global_ids)    send_ii(&o::td, /**/ &o::ti);
    if (multi_solvent) send_ii(&o::td, /**/ &o::tc);

    bulk(/**/ &o::q, &o::td);

    recv_pp(/**/ &o::td);
    if (global_ids)    recv_ii(/**/ &o::ti);
    if (multi_solvent) recv_ii(/**/ &o::tc);

    unpack();
   
    gather();
}
