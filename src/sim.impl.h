namespace sim {
/* see bund.cu for more sim:: functions */

void create_walls() {
  int nold = o::q.n;
  wall::gen_quants(&o::q.n, o::q.pp, &w::q); o::q.cells->build(o::q.pp, o::q.n);
  MSG("solvent particles survived: %d/%d", o::q.n, nold);
}

void create_solids() {
  cD2H(o::pp_hst, o::q.pp, o::q.n);
  rig::gen_quants(/*io*/ o::pp_hst, &o::q.n, /**/ &s::q);
  MC(l::m::Barrier(m::cart));
  cH2D(o::q.pp, o::pp_hst, o::q.n);
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
  if (o::q.n)           k_sim::clear_velocity<<<k_cnf(o::q.n)>>>(o::q.pp, o::q.n);  
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
  o::q.n = ic::gen(o::q.pp, /*w*/ o::pp_hst);
  o::q.cells->build(o::q.pp, o::q.n);
  get_ticketZ(o::q.pp, o::q.n, &o::tz);
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
    flu::get_ticketZ(o::q.pp, o::q.n, &o::tz);
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
  /*** flu::strt(&o::q.pp, &o::q.nn); ***/
  o::q.cells->build(/* io */ o::q.pp, o::q.n);

  /*** rbc::strt(&r::q); ***/
  dSync();

  /*** rig::strt(&s::q); ***/

  /*** wall::strt(&w::q); ***/

  /*T*/
  get_ticketZ(o::q.pp, o::q.n, &o::tz);
  if (walls) wall::gen_ticket(w::q, &w::t);
  flu::get_ticketZ(o::q.pp, o::q.n, &o::tz);

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
