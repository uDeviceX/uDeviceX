void dev2hst() { /* device to host  data transfer */
    int start = 0;
    cD2H(a::pp_hst + start, o::q.pp, o::q.n); start += o::q.n;
    if (solids0) {
        cD2H(a::pp_hst + start, s::q.pp, s::q.n); start += s::q.n;
    }
    if (rbcs) {
        cD2H(a::pp_hst + start, r::q.pp, r::q.n); start += r::q.n;
    }
}

void dump_part(int step) {
  cD2H(o::q.pp_hst, o::q.pp, o::q.n);
  if (global_ids) {
      cD2H(o::qi.ii_hst, o::qi.ii, o::q.n);
      bop::intdata(o::qi.ii_hst, o::q.n, "id_solvent", step);
  }
  if (multi_solvent) {
      cD2H(o::qc.ii_hst, o::qc.ii, o::q.n);
      bop::intdata(o::qc.ii_hst, o::q.n, "colors_solvent", step);
  }

  if (dump_forces) {
      cD2H(o::ff_hst, o::ff, o::q.n);
      bop::parts_forces(o::q.pp_hst, o::ff_hst, o::q.n, "solvent", step, /**/ &dumpt);
  } else {
      bop::parts(o::q.pp_hst, o::q.n, "solvent", step, /**/ &dumpt);
  }

  if(solids0) {
    cD2H(s::q.pp_hst, s::q.pp, s::q.n);
    if (dump_forces) {
        cD2H(s::ff_hst, s::ff, s::q.n);
        bop::parts_forces(s::q.pp_hst, s::ff_hst, s::q.n, "solid", step, /**/ &dumpt);
    } else {
        bop::parts(s::q.pp_hst, s::q.n, "solid", step, /**/ &dumpt);
    }
  }
}

void dump_rbcs() {
  static int id = 0;
  cD2H(a::pp_hst, r::q.pp, r::q.n);
  rbc_dump(a::pp_hst, r::q.tri_hst, r::q.nc, r::q.nv, r::q.nt, id++);
}

void dump_grid() {
    QQ qq; /* pack for io/field_dumps */
    NN nn;
    qq.o = o::q.pp; qq.s = s::q.pp; qq.r = r::q.pp;
    nn.o = o::q.n ; nn.s = s::q.n ;  nn.r = r::q.n;
    cD2H(a::pp_hst, o::q.pp, o::q.n);
    h5::dump(a::pp_hst, o::q.n);
}

void dump_diag_after(int it) { /* after wall */
    if (it % part_freq == 0)
    rig_dump(it, s::q.ss_dmp, s::t.ss_dmp, s::q.ns, m::coords);
}

void diag(int it) {
    int n = o::q.n + s::q.n + r::q.n; dev2hst();
    diagnostics(a::pp_hst, n, it);
}

void dump_strt_templ() { /* template dumps (wall, solid) */
    if (strt_dumps) {
        if (walls) wall::strt_dump_templ(w::q);
        if (solids) rig::strt_dump_templ(s::q);
    }
}

void dump_strt(int id) {
    flu::strt_dump(id, o::q);
    if (global_ids)    flu::strt_dump_ii("id",     id, o::qi, o::q.n);
    if (multi_solvent) flu::strt_dump_ii("colors", id, o::qc, o::q.n);
    if (rbcs)       rbc::strt_dump(id, r::q);
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
}
