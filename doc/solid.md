# decl

* `ff`: a usual variable in `sim::`

* [p] : `pp[]`, `npp`, `i_pp[]`, `ss[]`, `ns`
or
* [z] : no
* [w] : all others (see bellow)

## plan for sim::
   [p] = read/create(...)
   [w] = solid::allocate([p])

   [p] = solid::func([p], [w])

   **read/write**
   [p] = not_solid::func([p])

   **read**
   [p] = not_solid::func([p])

   solid::deallocate([w])
   deallocate([p])

## plan for solid::
	solid::func([p], [w])
   are not allowed to keep anything in [w]

## sim.impl
* `remove_solids_from_wall` [-]
uses `s::m_dev.nv`

* `set_ids_solids`

* `wall::interactions` [+]
uses [p] and `ff`

* `k_sim::clear_velocity` [+]
uses [p]

* `update_solid0()`
does not use anything ?

## impl
* `load_solid_mesh(const char *fname)` [-]
allocates `m_dev.tt`, `m_dev.tt`

* `allocate`
* `allocate_tcells`
* `deallocate`

* `create(Particle *opp, int *on)`
does something wired also calls `load_solid_mesh`?

## ic
* `set_ids(const int ns, Solid *ss_hst)`
* `void init(const char *fname, const Mesh m, /**/
	 int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp)`

## [w] variables
> Not all [p] variables removed.
> 14:58 if DEVICE_SOLID is not defined, ss_hst is in [p]
> 14:58 otherwise, it may be in zip or [w] and ss_dev is in [p]

``` C
 *ff
 pp_hst[MAX_PART_NUM] /* Solid pp on host */
 ff_hst[MAX_PART_NUM] /* Solid ff on host */
 m_hst /* mesh of solid on host */
 m_dev /* mesh of solid on device */
 [t]riangle [c]ells [s]tarts / [c]ounts / [i]ds */
 *tcs_hst, *tcc_hst, *tci_hst /* [t]riangle cell-lists on host */
 *tcs_dev, *tcc_dev, *tci_dev /* [t]riangle cell-lists on device */
 *bboxes_hst /* [b]ounding [b]oxes of solid mesh on host */
 *bboxes_dev /* [b]ounding [b]oxes of solid mesh on device */
 *i_pp_hst, *i_pp_dev /* particles representing vertices of ALL meshes of solid [i]nterfaces */
 *i_pp_bb_hst, *i_pp_bb_dev /* buffers for BB multi-nodes */
 nps /* number of particles per solid */
 *ss_hst /* solid infos on host */
 *ss_dev /* solid infos on device */
 *ss_bb_hst /* solid buffer for bounce back, host */
 *ss_bb_dev /* solid buffer for bounce back, device */
 buffers of solids for dump this is needed because we dump the BB F and T separetely */
 *ss_dmphst, *ss_dmpbbhst
 rr0_hst[3*MAX_PSOLID_NUM] /* initial positions same for all solids */
 *rr0
```
