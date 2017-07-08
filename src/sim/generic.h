void distr_rbc() {
  rdstr::extent(r::q.pp, r::q.nc, r::q.nv);
  dSync();
  rdstr::pack_sendcnt(r::q.pp, r::q.nc, r::q.nv);
  r::q.nc = rdstr::post(r::q.nv); r::q.n = r::q.nc * r::q.nv;
  rdstr::unpack(r::q.pp, r::q.nv);
}

template <typename T>
void remove(T *data, int nv, int *e, int nc) {
  int c; /* c: cell */
  for (c = 0; c < nc; c++) cA2A(data + nv*c, data + nv*e[c], nv);
}

void remove_rbcs() {
  int stay[MAX_CELL_NUM];
  int nc0;
  r::q.nc = sdf::who_stays(w::qsdf, r::q.pp, r::q.n, nc0 = r::q.nc, r::q.nv, /**/ stay);
  r::q.n = r::q.nc * r::q.nv;
  Cont::remove(r::q.pp, r::q.nv, stay, r::q.nc);
  MSG("%d/%d RBCs survived", r::q.nc, nc0);
}

void remove_solids() {
  int stay[MAX_SOLIDS];
  int ns0;
  int nip = s::q.ns * s::q.m_dev.nv;
  s::q.ns = sdf::who_stays(w::qsdf, s::q.i_pp, nip, ns0 = s::q.ns, s::q.m_dev.nv, /**/ stay);
  s::q.n  = s::q.ns * s::q.nps;
  Cont::remove(s::q.pp,       s::q.nps,      stay, s::q.ns);
  Cont::remove(s::q.pp_hst,   s::q.nps,      stay, s::q.ns);

  Cont::remove(s::q.ss,       1,           stay, s::q.ns);
  Cont::remove(s::q.ss_hst,   1,           stay, s::q.ns);

  Cont::remove(s::q.i_pp,     s::q.m_dev.nv, stay, s::q.ns);
  Cont::remove(s::q.i_pp_hst, s::q.m_hst.nv, stay, s::q.ns);
  MSG("sim.impl: %d/%d Solids survived", s::q.ns, ns0);
}

