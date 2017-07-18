void step(float driving_force0, bool wall0, int it) {
    assert(o::q.n <= MAX_PART_NUM);
    assert(r::q.n <= MAX_PART_NUM);

    odstr::post_recv(&o::td);
    odstr::pack(&o::q, &o::td);
    odstr::send(&o::td);
    odstr::bulk(&o::q, &o::td);
    odstr::recv(&o::td);
    //odstr::unpack(&o::q, &o::td, &o::tu, &o::tz, &o::w);

    odstr::unpack_pp(&o::q, &o::td, &o::tu, &o::w);
    if (global_ids) odstr::unpack_ii(&o::td, &o::tu);
    odstr::gather_pp(&o::td, /**/ &o::q, &o::tu, &o::tz);
    if (global_ids) odstr::gather_ii(&o::tu, /**/ &o::q);
    
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
