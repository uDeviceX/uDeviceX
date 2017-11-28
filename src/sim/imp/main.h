enum {
    NCELLSWALL =
    (XS + 2*XWM) *
    (YS + 2*YWM) *
    (ZS + 2*ZWM)
};

enum {
    MAXNWALL = NCELLSWALL * numberdensity
};

void gen() { /* generate */
    run_eq(wall_creation);
    if (walls) {
        dSync();
        UC(sdf::ini(m::cart, &w::qsdf));
        MC(m::Barrier(m::cart));
        inter::create_walls(MAXNWALL, w::qsdf, /*io*/ &flu.q, /**/ &w::q);
    }
    inter::freeze(m::cart, w::qsdf, /*io*/ &flu.q, /**/ &s::q, &rbc.q);
    clear_vel();

    if (multi_solvent) {
        Particle *pp = flu.q.pp;
        int n = flu.q.n;
        int *cc = flu.q.cc;
        Particle *pp_hst = a::pp_hst;
        int *cc_hst = flu.q.cc_hst;
        inter::color_dev(pp, n, /*o*/ cc, /*w*/ pp_hst, cc_hst);
    }
}

void sim_gen() {
    flu::gen_quants(&flu.q);
    flu::build_cells(&flu.q);
    if (global_ids)    flu::gen_ids  (m::cart, flu.q.n, &flu.q);
    if (rbcs) {
        rbc::main::gen_quants(m::cart, "rbc.off", "rbcs-ic.txt", /**/ &rbc.q);
        rbc::force::gen_ticket(rbc.q, &rbc.tt);

        if (multi_solvent) gen_colors(&rbc, &colorer, /**/ &flu);
    }
    MC(m::Barrier(m::cart));

    long nsteps = (long)(tend / dt);
    MSG("will take %ld steps", nsteps);
    if (walls || solids) {
        solids0 = false;  /* global */
        gen();
        dSync();
        if (walls && w::q.n) wall::gen_ticket(w::q, &w::t);
        solids0 = solids;
        if (rbcs && multi_solvent) gen_colors(&rbc, &colorer, /**/ &flu);
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
    flu::strt_quants(restart::BEGIN, &flu.q);
    flu::build_cells(&flu.q);

    if (rbcs) rbc::main::strt_quants("rbc.off", restart::BEGIN, &rbc.q);
    dSync();

    if (solids) rig::strt_quants(restart::BEGIN, &s::q);

    if (walls) wall::strt_quants(MAXNWALL, &w::q);

    /*T*/
    if (rbcs)            rbc::force::gen_ticket(rbc.q, &rbc.tt);
    if (walls && w::q.n) wall::gen_ticket(w::q, &w::t);

    MC(m::Barrier(m::cart));
    if (walls) {
        dSync();
        UC(sdf::ini(m::cart, &w::qsdf));
        MC(m::Barrier(m::cart));
    }

    solids0 = solids;

    MSG("will take %ld steps", nsteps - wall_creation);
    run(wall_creation, nsteps);

    if (strt_dumps) dump_strt(restart::FINAL);
}
