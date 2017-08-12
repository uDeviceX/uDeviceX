/* see bund.cu for more sim:: functions */

void create_walls() {
    int nold = o::q.n;
    wall::gen_quants(w::qsdf, /**/ &o::q.n, o::q.pp, &w::q);
    o::q.cells->build(o::q.pp, o::q.n);
    MSG("solvent particles survived: %d/%d", o::q.n, nold);
}

void create_solids() {
    cD2H(o::q.pp_hst, o::q.pp, o::q.n);
    rig::gen_quants(/*io*/ o::q.pp_hst, &o::q.n, /**/ &s::q);
    MC(l::m::Barrier(l::m::cart));
    cH2D(o::q.pp, o::q.pp_hst, o::q.n);
    MC(l::m::Barrier(l::m::cart));
    MSG("created %d solids.", s::q.ns);
}

void freeze() {
    MC(l::m::Barrier(l::m::cart));
    if (solids)           create_solids();
    if (walls && rbcs  )  remove_rbcs();
    if (walls && solids)  remove_solids();
    if (solids)           rig::set_ids(s::q);
}

void clear_vel() {
    KL(dev::clear_vel, (k_cnf(o::q.n)), (o::q.pp, o::q.n));
    if (solids) KL(dev::clear_vel, (k_cnf(s::q.n)), (s::q.pp, s::q.n));
    if (rbcs  ) KL(dev::clear_vel, (k_cnf(r::q.n)), (r::q.pp, r::q.n));
}

void gen() { /* generate */
    run_eq(wall_creation);
    if (walls) {
        dSync();
        sdf::ini(&w::qsdf);
        MC(l::m::Barrier(l::m::cart));
        create_walls();
    }
    freeze();
    clear_vel();
}

void sim_gen() {
    flu::gen_quants(&o::q);
    if (global_ids)    flu::gen_ids  (o::q.n, &o::qi);
    if (multi_solvent) flu::gen_tags0(o::q.n, &o::qt);
    o::q.cells->build(o::q.pp, o::q.n);
    flu::get_ticketZ(o::q, &o::tz);
    flu::get_ticketRND(&o::trnd);
    if (rbcs) {
        rbc::gen_quants("rbc.off", "rbcs-ic.txt", /**/ &r::q);
        rbc::gen_ticket(r::q, &r::tt);

        if (multi_solvent) gen_tags();
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
        if (rbcs && multi_solvent) gen_tags();
        run(wall_creation, nsteps);
    } else {
        solids0 = solids;
        run(            0, nsteps);
    }
    /* final strt dump*/
    if (strt_dumps) dump_strt(restart::FINAL);
}

void sim_strt() {
    long nsteps = (int)(tend / dt);
    
    /*Q*/
    flu::strt_quants(restart::BEGIN, &o::q);
    if (global_ids)    flu::strt_ii("id",   restart::BEGIN, &o::qi);
    if (multi_solvent) flu::strt_ii("tags", restart::BEGIN, &o::qt);
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
