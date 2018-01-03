namespace flu {

struct Quants {
    Particle *pp, *pp0;    /* particles on device  */
    int       n;           /* particle number      */
    clist::Clist cells;    /* cell lists           */
    clist::Map  mcells;    /* cell lists map       */
    Particle *pp_hst;      /* particles on host    */

    /* optional data */

    int *ii, *ii0;  /* global ids on device */
    int *ii_hst;    /* global ids on host   */

    int *cc, *cc0;  /* colors on device */
    int *cc_hst;    /* colors on host   */    
}; 

void ini(Quants *q);
void fin(Quants *q);

void gen_quants(Coords coords, Quants *q);
void gen_ids(MPI_Comm comm, const int n, Quants *q);

void strt_quants(Coords coords, const int id, Quants *q);
void strt_dump(Coords coords, const int id, const Quants q);

/* build cells only from one array of particles fully contained in the domain */
/* warning: this will delete particles which are outside                      */
void build_cells(/**/ Quants *q);

} // flu
