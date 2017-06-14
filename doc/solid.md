# solid

## decl

* `ff`: is a variable in `sim::`

* [p] : `pp[]`, `npp`

* [z] : `i_pp[]`, `ss[]`, `ns`
or
* [z] : no

### sim.impl
* `remove_solids_from_wall`
* `set_ids_solids`
* `k_sim::clear_velocity`

### impl
* `load_solid_mesh`
* `allocate`
* `allocate_tcells`
* `deallocate`
* `create(Particle *opp, int *on)`
* `load_solid_mesh(const char *fname)`
* `set_ids`

### ic
* `set_ids(const int ns, Solid *ss_hst)`
* `void init(const char *fname, const Mesh m, /**/ int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp)`

* [w] : the rest of of variables [src/s/decl.h]
pp_hst[MAX_PART_NUM]
ff_hst[MAX_PART_NUM]
m_hst
m_dev
*tcs_hst,
*tcs_dev,
*bboxes_hst
*bboxes_dev
*i_pp_hst,
*i_pp_bb_hst,
*ss_hst
*ss_dev
*ss_bb_hst
*ss_bb_dev
*ss_dmphst,
rr0_hst[3*MAX_PSOLID_NUM]
*rr0
