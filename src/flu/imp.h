namespace flu {

struct FluQuants {
    Particle *pp, *pp0;    /* particles on device  */
    int       n;           /* particle number      */
    Clist      cells;      /* cell lists           */
    ClistMap *mcells;      /* cell lists map       */
    Particle *pp_hst;      /* particles on host    */

    /* optional data */

    int *ii, *ii0;  /* global ids on device */
    int *ii_hst;    /* global ids on host   */

    int *cc, *cc0;  /* colors on device */
    int *cc_hst;    /* colors on host   */    
}; 

void ini(FluQuants *q);
void fin(FluQuants *q);

void gen_quants(Coords coords, FluQuants *q);
void gen_ids(MPI_Comm comm, const int n, FluQuants *q);

void strt_quants(Coords coords, const int id, FluQuants *q);
void strt_dump(Coords coords, const int id, const FluQuants q);

/* build cells only from one array of particles fully contained in the domain */
/* warning: this will delete particles which are outside                      */
void build_cells(/**/ FluQuants *q);

} // flu
