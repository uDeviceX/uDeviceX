void gen() { /* generate */
    run_eq(wall_creation);
    if (walls) {
        dSync();
        sdf::ini(&w::qsdf);
        MC(l::m::Barrier(l::m::cart));
        inter::create_walls(w::qsdf, /*io*/ &o::q, /**/ &w::q);
    }
    inter::freeze(w::qsdf, /*io*/ &o::q, /**/ &s::q, &r::q);
    clear_vel();
}

void sim_gen() {
    flu::gen_quants(&o::q);
    if (global_ids)    flu::gen_ids  (o::q.n, &o::qi);
    if (multi_solvent) flu::gen_tags0(o::q.n, &o::qc);
    o::q.cells->build(o::q.pp, o::q.n);
    flu::get_ticketZ(o::q, &o::tz);
    flu::get_ticketRND(&o::trnd);
    if (rbcs) {
        rbc::gen_quants("rbc.off", "rbcs-ic.txt", /**/ &r::q);
        rbc::gen_ticket(r::q, &r::tt);

        if (multi_solvent) gen_colors();
    }
    MC(l::m::Barrier(l::m::cart));
  
    long nsteps = (long)(tend / dt);
    MSG("will take %ld steps", nsteps);
    if (walls || solids) {
        solids0 = false;  /* global */
        gen();
        dSync();
        if (walls && w::q.n) wall::gen_ticket(w::q, &w::t);
        flu::get_ticketZ(o::q, &o::tz);
        flu::get_ticketRND(&o::trnd);
        solids0 = solids;
        if (rbcs && multi_solvent) gen_colors();
        run(wall_creation, nsteps);
    } else {
        solids0 = solids;
        run(            0, nsteps);
    }
    /* final strt dump*/
    if (strt_dumps) dump_strt(restart::FINAL);
}

void sim_strt() {
    long nsteps = (long)(tend / dt);
    
    /*Q*/
    flu::strt_quants(restart::BEGIN, &o::q);
    if (global_ids)    flu::strt_ii("id",     restart::BEGIN, &o::qi);
    if (multi_solvent) flu::strt_ii("colors", restart::BEGIN, &o::qc);
    o::q.cells->build(/* io */ o::q.pp, o::q.n);

    if (rbcs) rbc::strt_quants("rbc.off", restart::BEGIN, &r::q);
    dSync();

    if (solids) rig::strt_quants(restart::BEGIN, &s::q);

    if (walls) wall::strt_quants(&w::q);

    /*T*/
    flu::get_ticketZ(o::q, &o::tz);
    flu::get_ticketRND(&o::trnd);
    if (rbcs)            rbc::gen_ticket(r::q, &r::tt);
    if (walls && w::q.n) wall::gen_ticket(w::q, &w::t);

    MC(l::m::Barrier(l::m::cart));
    if (walls) {
        dSync();
        sdf::ini(&w::qsdf);
        MC(l::m::Barrier(l::m::cart));
    }

    solids0 = solids;

    MSG("will take %ld steps", nsteps - wall_creation);
    run(wall_creation, nsteps);
    
    /* final strt dump*/
    if (strt_dumps) dump_strt(restart::FINAL);
}
