namespace sim {
/* see bund.cu for more sim:: functions */

void create_walls() {
  int nold = o::n;
  wall::gen_quants(&o::n, o::pp, &w::q); o::cells->build(o::pp, o::n);
  MSG("solvent particles survived: %d/%d", o::n, nold);
}

void create_solids() {
  cD2H(o::pp_hst, o::pp, o::n);
  rig::gen_quants(/*io*/ o::pp_hst, &o::n, /**/ &s::q);
  MC(l::m::Barrier(m::cart));
  cH2D(o::pp, o::pp_hst, o::n);
  MC(l::m::Barrier(m::cart));
  MSG("created %d solids.", s::q.ns);
}

void freeze() {
  MC(MPI_Barrier(m::cart));
  if (solids)           create_solids();
  if (walls && rbcs  )  remove_rbcs();
  if (walls && solids)  remove_solids();
  if (solids)           rig::set_ids(s::q);
}

void clear_velocity() {
  if (o::n)             k_sim::clear_velocity<<<k_cnf(o::n)>>>(o::pp, o::n);  
  if (solids && s::q.n) k_sim::clear_velocity<<<k_cnf(s::q.n)>>>(s::q.pp, s::q.n);
  if (rbcs   && r::q.n) k_sim::clear_velocity<<<k_cnf(r::q.n)>>>(r::q.pp, r::q.n);
}

void gen() { /* generate */
  run_eq(wall_creation);
  if (walls) {
    dSync();
    sdf::ini();
    create_walls();
  }
  freeze();
  clear_velocity();
}

void sim_gen() {
  o::n = ic::gen(o::pp, /*w*/ o::pp_hst);
  o::cells->build(o::pp, o::n);
  get_ticketZ(o::pp, o::n, &o::tz);
  if (rbcs) {
      rbc::gen_quants("rbc.off", "rbcs-ic.txt", /**/ &r::q);
      rbc::gen_ticket(r::q, &r::tt);
  }
  MC(MPI_Barrier(m::cart));
  
  long nsteps = (int)(tend / dt);
  MSG0("will take %ld steps", nsteps);
  if (walls || solids) {
    solids0 = false;  /* global */
    gen();
    dSync();
    if (walls) wall::gen_ticket(w::q, &w::t);
    flu::get_ticketZ(o::pp, o::n, &o::tz);
    solids0 = solids;
    run(wall_creation, nsteps);
  } else {
    solids0 = solids;
    run(            0, nsteps);
  }
}

void sim_strt() {
  long nsteps = (int)(tend / dt);

  /*Q*/
  /*** flu::strt(&o::pp, &o::nn); ***/
  o::cells->build(/* io */ o::pp, o::n);

  /*** rbc::strt(&r::q); ***/
  dSync();

  /*** rig::strt(&s::q); ***/

  /*** wall::strt(&w::q); ***/

  /*T*/
  get_ticketZ(o::pp, o::n, &o::tz);
  if (walls) wall::gen_ticket(w::q, &w::t);
  flu::get_ticketZ(o::pp, o::n, &o::tz);

  MC(MPI_Barrier(m::cart));
  if (walls) {
    dSync();
    sdf::ini();
    create_walls();
  }

  solids0 = solids;
  run(wall_creation, nsteps);
}

}
