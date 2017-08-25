static void gather() {
    gather_pp(&o::td, /**/ &o::q, &o::tu, &o::tz);
    if (global_ids)    gather_ii(o::q.n, &o::tu, &o::tui, /**/ &o::qi);
    if (multi_solvent) gather_ii(o::q.n, &o::tu, &o::tut, /**/ &o::qt);
}

static void pack() {
    pack_pp(&o::q, /**/ &o::td);
    if (global_ids)    pack_ii(o::q.n, &o::qi, &o::td, /**/ &o::ti);
    if (multi_solvent) pack_ii(o::q.n, &o::qt, &o::td, /**/ &o::tt);
}

static void unpack() {
    unpack_pp(&o::td, /**/ &o::q, &o::tu, /*w*/ &o::w);
    if (global_ids)    unpack_ii(&o::td, &o::ti, /**/ &o::tui);
    if (multi_solvent) unpack_ii(&o::td, &o::tt, /**/ &o::tut);
}

void odstr() {
    using namespace odstr;

    assert(o::q.n <= MAX_PART_NUM);
    assert(r::q.n <= MAX_PART_NUM);

    post_recv_pp(/**/ &o::td);
    if (global_ids)    post_recv_ii(&o::td, /**/ &o::ti);
    if (multi_solvent) post_recv_ii(&o::td, /**/ &o::tt);

    pack();

    send_pp(/**/ &o::td);
    if (global_ids)    send_ii(&o::td, /**/ &o::ti);
    if (multi_solvent) send_ii(&o::td, /**/ &o::tt);

    bulk(/**/ &o::q, &o::td);

    recv_pp(/**/ &o::td);
    if (global_ids)    recv_ii(/**/ &o::ti);
    if (multi_solvent) recv_ii(/**/ &o::tt);

    unpack();
   
    gather();

    dbg::check_pp(o::q.pp, o::q.n, "flu: distr pp");
}
