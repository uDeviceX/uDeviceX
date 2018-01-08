struct PidVCont {
    int3 L;                   /* subdomain size                               */
    float3 target, current;   /* target and current average velocities        */
    float Kp, Ki, Kd, factor; /* parameters of the pid controller             */
    float3 olde, sume;        /* previous error, sum of all previous errors   */
    float3 f;                 /* force estimate                               */
    long nsamples;            /* number of "pending" avg on grid              */
    long totncells;           /* total number of cells                        */
    
    float3 *gridvel;          /* average velocity per grid point  */
    float3 *avgvel;           /* chunk sums (pinned memory)       */
    float3 *davgvel;          /* device pointer of the above      */

    MPI_Comm comm;

    FILE *fdump;              /* output file for logging info     */
};

void vcont_ini(MPI_Comm comm, int3 L, float3 vtarget, float factor, /**/ PidVCont *c);
void vcont_fin(/**/ PidVCont *cont);

void   vcont_sample(Coords coords, int n, const Particle *pp, const int *starts, const int *counts, /**/ PidVCont *c);
float3 vcont_adjustF(/**/ PidVCont *c);
void   vcont_log(const PidVCont *c);
