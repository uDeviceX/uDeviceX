void odstr() {
    assert(o::q.n <= MAX_PART_NUM);
    assert(r::q.n <= MAX_PART_NUM);

    odstr::post_recv_pp(/**/ &o::td);
    if (global_ids)    odstr::post_recv_ii(&o::td, /**/ &o::ti);
    if (multi_solvent) odstr::post_recv_ii(&o::td, /**/ &o::tt);
    
    odstr::pack_pp(&o::q, /**/ &o::td);
    if (global_ids)    odstr::pack_ii(o::q.n, &o::qi, &o::td, /**/ &o::ti);
    if (multi_solvent) odstr::pack_ii(o::q.n, &o::qt, &o::td, /**/ &o::tt);

    odstr::send_pp(/**/ &o::td);
    if (global_ids)    odstr::send_ii(&o::td, /**/ &o::ti);
    if (multi_solvent) odstr::send_ii(&o::td, /**/ &o::tt);
    
    odstr::bulk(/**/ &o::q, &o::td);

    odstr::recv_pp(/**/ &o::td);
    if (global_ids)    odstr::recv_ii(/**/ &o::ti);
    if (multi_solvent) odstr::recv_ii(/**/ &o::tt);
    
    odstr::unpack_pp(&o::td, /**/ &o::q, &o::tu, /*w*/ &o::w);
    if (global_ids)    odstr::unpack_ii(&o::td, &o::ti, /**/ &o::tui);
    if (multi_solvent) odstr::unpack_ii(&o::td, &o::tt, /**/ &o::tut);
    
    odstr::gather_pp(&o::td, /**/ &o::q, &o::tu, &o::tz);
    if (global_ids)    odstr::gather_ii(o::q.n, &o::tu, &o::tui, /**/ &o::qi);
    if (multi_solvent) odstr::gather_ii(o::q.n, &o::tu, &o::tut, /**/ &o::qt);
}
