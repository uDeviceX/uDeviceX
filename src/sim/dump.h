namespace sim {
void dev2hst() { /* device to host  data transfer */
    int start = 0;
    cD2H(a::pp_hst + start, o::pp, o::n); start += o::n;
    if (solids0) {
        cD2H(a::pp_hst + start, s::q.pp, s::q.n); start += s::q.n;
    }
    if (rbcs) {
        cD2H(a::pp_hst + start, r::q.pp, r::q.n); start += r::q.n;
    }
}

void dump_part(int step) {
  cD2H(o::pp_hst, o::pp, o::n);
  dump::parts(o::pp_hst, o::n, "solvent", step);

  if(solids0) {
    cD2H(s::q.pp_hst, s::q.pp, s::q.n);
    dump::parts(s::q.pp_hst, s::q.n, "solid", step);
  }
}

void dump_rbcs() {
  static int id = 0;
  cD2H(a::pp_hst, r::q.pp, r::q.n);
  rbc_dump(r::q.nc, a::pp_hst, r::q.tri_hst, r::q.nv, r::q.nt, id++);
}

void dump_grid() {
  cD2H(a::pp_hst, o::pp, o::n);
  dump_field->dump(a::pp_hst, o::n);
}

void diag(int it) {
    int n = o::n + s::q.n + r::q.n; dev2hst();
    diagnostics(a::pp_hst, n, it);
}
}
