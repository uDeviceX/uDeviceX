enum {
    TYPE_NONE,
    TYPE_CART,
    TYPE_RAD
};

// empty for now
struct TCart {};
struct TRad {};

/* parameters for transformation */
union Trans {
    TCart cart;
    TRad  rad;
};


struct Sampler {
    int3 L;                   /* subdomain size                                  */
    long nsamples;            /* number of "pending" avg on grid                 */
    
    float3 *gridvel;          /* sum of velocity per grid point                  */
    int    *gridnum;          /* sum of number of particles per grid point       */
    float3 *totvel;           /* chunk sums (pinned memory)                      */
    int    *totnum;
    float3 *dtotvel;          /* device pointer of the above                     */
    int    *dtotnum;
};

struct State {
    float3 cur;               /* current average velocity                        */
    float3 olde, sume;        /* previous error, sum of all previous errors      */
    float3 f;                 /* force estimate                                  */
};

struct Param {
    float3 target;            /* target average velocity                         */
    float Kp, Ki, Kd;         /* parameters of the pid controller                */
};

/* pid velocity controller */
struct PidVCont {
    Param param;
    State state;
    Sampler sampler;          /* helper to average velocity on device            */
    
    MPI_Comm comm;

    FILE *fdump;              /* output file for logging info                    */

    Trans trans;              /* transformation of the velocity before averaging */
    int type;                 /* type of the transformation                      */
};
