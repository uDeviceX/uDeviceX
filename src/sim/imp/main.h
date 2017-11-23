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
        sdf::ini(&w::qsdf);
        MC(m::Barrier(m::cart));
        inter::create_walls(MAXNWALL, w::qsdf, /*io*/ &o::q, /**/ &w::q);
    }
    inter::freeze(w::qsdf, /*io*/ &o::q, /**/ &s::q, &r::q);
    clear_vel();

    if (multi_solvent) {
        Particle *pp = o::q.pp;
        int n = o::q.n;
        int *cc = o::q.cc;
        Particle *pp_hst = a::pp_hst;
        int *cc_hst = o::q.cc_hst;
        inter::color_dev(pp, n, /*o*/ cc, /*w*/ pp_hst, cc_hst);
    }
}

void sim_gen() {
    flu::gen_quants(&o::q);
    flu::build_cells(&o::q);
    if (global_ids)    flu::gen_ids  (o::q.n, &o::q);
    if (rbcs) {
        rbc::main::gen_quants("rbc.off", "rbcs-ic.txt", /**/ &r::q);
        rbc::force::gen_ticket(r::q, &r::tt);

        if (multi_solvent) gen_colors(&colorer);
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
        if (rbcs && multi_solvent) gen_colors(&colorer);
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
    flu::build_cells(&o::q);

    if (rbcs) rbc::main::strt_quants("rbc.off", restart::BEGIN, &r::q);
    dSync();

    if (solids) rig::strt_quants(restart::BEGIN, &s::q);

    if (walls) wall::strt_quants(MAXNWALL, &w::q);

    /*T*/
    if (rbcs)            rbc::force::gen_ticket(r::q, &r::tt);
    if (walls && w::q.n) wall::gen_ticket(w::q, &w::t);

    MC(m::Barrier(m::cart));
    if (walls) {
        dSync();
        sdf::ini(&w::qsdf);
        MC(m::Barrier(m::cart));
    }

    solids0 = solids;

    MSG("will take %ld steps", nsteps - wall_creation);
    run(wall_creation, nsteps);

    if (strt_dumps) dump_strt(restart::FINAL);
}
