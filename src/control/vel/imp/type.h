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
