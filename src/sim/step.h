void odstr() {
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

void step(float driving_force0, bool wall0, int it) {
    assert(o::q.n <= MAX_PART_NUM);
    assert(r::q.n <= MAX_PART_NUM);

    odstr();

    if (solids0) distr_solid();
    if (rbcs)    distr_rbc();
    forces(wall0);
    dump_diag0(it);
    if (wall0 || solids0) dump_diag_after(it);
    body_force(driving_force0);
    update_solvent();
    if (solids0) update_solid();
    if (rbcs)    update_rbc();
    if (wall0) bounce();
    if (sbounce_back && solids0) bounce_solid(it);
}
