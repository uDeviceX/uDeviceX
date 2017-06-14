# solid

## decl

* `ff`: is a variable in `sim::`

* [p] : `pp[]`, `npp`, `i_pp[]`, `ss[]`, `ns`
or
* [z] : no

### sim.impl
* `remove_solids_from_wall` [-]
uses `s::m_dev.nv`

* `set_ids_solids`

* `wall::interactions` [+]
uses [p] and `ff`

* `k_sim::clear_velocity` [+]
uses [p]

* `update_solid0()`
does not use anything ?

### impl
* `load_solid_mesh(const char *fname)` [-]
allocates `m_dev.tt`, `m_dev.tt`

* `allocate`
* `allocate_tcells`
* `deallocate`

* `create(Particle *opp, int *on)`
does something wired also calls `load_solid_mesh`?

### ic
* `set_ids(const int ns, Solid *ss_hst)`
* `void init(const char *fname, const Mesh m, /**/
     int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp)`

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
