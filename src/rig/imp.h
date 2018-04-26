// tag::quants[]
struct RigQuants {
    int n, ns, nps;              /* number of particles (total), solid, particle per solid        */
    Particle *pp_hst, *pp;       /* particles on hst and device                                   */
    Solid    *ss_hst, *ss;       /* rigid strutures                                               */
    float   *rr0_hst, *rr0;      /* frozen particle templates                                     */

    /* mesh related quantities */
    int nt, nv;                   /* number of [t]riangles and [v]ertices                          */
    int4 *dtt;                    /* triangle indices on device                                    */
    float *dvv;                   /* vertices on device (template)                                 */
    Particle *i_pp_hst, *i_pp;    /* particles representing all meshes of all solids of that node  */

    Solid *ss_dmp, *ss_dmp_bb;

    long maxp; /* maximum particle number */
};
// end::quants[]

struct Coords;
struct RigPinInfo;
struct MeshRead;

// tag::mem[]
void rig_ini(long maxs, long maxp, const MeshRead*, RigQuants*);
void rig_fin(RigQuants *q);
// end::mem[]

// tag::gen[]
void rig_gen_mesh(const Coords*, MPI_Comm, const MeshRead*, const char *ic, /**/ RigQuants *);
void rig_gen_freeze(const Coords*, bool empty_pp, int numdensity, float rig_mass, const RigPinInfo*, MPI_Comm, const MeshRead*, Particle *opp, int *on, RigQuants *q);
void rig_strt_quants(MPI_Comm, const MeshRead*, const char *base, const int id, RigQuants *q);
// end::gen[]

// tag::io[]
void rig_strt_dump_templ(MPI_Comm, const char *base, const RigQuants *q);         // <1>
void rig_strt_dump(MPI_Comm, const char *base, const int id, const RigQuants *q); // <2>
// tag::io[]
