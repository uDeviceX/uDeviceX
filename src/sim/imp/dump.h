void dev2hst() { /* device to host  data transfer */
    int start = 0;
    cD2H(a::pp_hst + start, flu.q.pp, flu.q.n); start += flu.q.n;
    if (solids0) {
        cD2H(a::pp_hst + start, s::q.pp, s::q.n); start += s::q.n;
    }
    if (rbcs) {
        cD2H(a::pp_hst + start, r::q.pp, r::q.n); start += r::q.n;
    }
}

void dump_part(int step) {
    cD2H(flu.q.pp_hst, flu.q.pp, flu.q.n);
    if (global_ids) {
        cD2H(flu.q.ii_hst, flu.q.ii, flu.q.n);
        bop::ids(m::cart, flu.q.ii_hst, flu.q.n, "id_solvent", step);
    }
    if (multi_solvent) {
        cD2H(flu.q.cc_hst, flu.q.cc, flu.q.n);
        bop::colors(m::cart, flu.q.cc_hst, flu.q.n, "colors_solvent", step);
    }

    if (force_dumps) {
        cD2H(flu.ff_hst, flu.ff, flu.q.n);
        bop::parts_forces(m::cart, flu.q.pp_hst, flu.ff_hst, flu.q.n, "solvent", step, /**/ &dumpt);
    } else {
        bop::parts(m::cart, flu.q.pp_hst, flu.q.n, "solvent", step, /**/ &dumpt);
    }

    if(solids0) {
        cD2H(s::q.pp_hst, s::q.pp, s::q.n);
        if (force_dumps) {
            cD2H(s::ff_hst, s::ff, s::q.n);
            bop::parts_forces(m::cart, s::q.pp_hst, s::ff_hst, s::q.n, "solid", step, /**/ &dumpt);
        } else {
            bop::parts(m::cart, s::q.pp_hst, s::q.n, "solid", step, /**/ &dumpt);
        }
    }
}

void dump_rbcs() {
    static int id = 0;
    cD2H(a::pp_hst, r::q.pp, r::q.n);
    io::mesh::rbc(m::cart, a::pp_hst, r::q.tri_hst, r::q.nc, r::q.nv, r::q.nt, id++);
}

void dump_rbc_coms() {
    static int id = 0;
    int nc = r::q.nc;
    rbc::com::get(r::q.nc, r::q.nv, r::q.pp, /**/ &r::com);
    dump_com(m::cart, id++, nc, r::q.ii, r::com.hrr);
}

void dump_grid() {
    QQ qq; /* pack for io/field_dumps */
    NN nn;
    qq.o = flu.q.pp; qq.s = s::q.pp; qq.r = r::q.pp;
    nn.o = flu.q.n ; nn.s = s::q.n ;  nn.r = r::q.n;
    fields_grid(m::cart, qq, nn, /*w*/ a::pp_hst);
}

void dump_diag_after(int it, bool wall0, bool solid0) { /* after wall */
    if (solid0 && it % part_freq == 0) {
        static int id = 0;
        rig_dump(it, s::q.ss_dmp, s::q.ss_dmp_bb, s::q.ns, m::coords);

        cD2H(a::pp_hst, s::q.i_pp, s::q.ns * s::q.nv);
        io::mesh::rig(m::cart, a::pp_hst, s::q.htt, s::q.ns, s::q.nv, s::q.nt, id++);
    }
}

void diag(int it) {
    int n = flu.q.n + s::q.n + r::q.n; dev2hst();
    diagnostics(m::cart, n, a::pp_hst, it);
}

void dump_strt_templ() { /* template dumps (wall, solid) */
    if (strt_dumps) {
        if (walls) wall::strt_dump_templ(w::q);
        if (solids) rig::strt_dump_templ(s::q);
    }
}

void dump_strt(int id) {
    flu::strt_dump(id, flu.q);
    if (rbcs)       rbc::main::strt_dump(id, r::q);
    if (solids)     rig::strt_dump(id, s::q);
}

void dump_diag0(int it) { /* generic dump */
    if (it % part_freq  == 0) {
        if (part_dumps) dump_part(it);
        if (rbcs)       dump_rbcs();
        diag(it);
    }
    if (field_dumps && it % field_freq == 0) dump_grid();
    if (strt_dumps  && it % strt_freq == 0)  dump_strt(it / strt_freq);
    if (rbc_com_dumps && it % rbc_com_freq == 0) dump_rbc_coms();
}
