struct WallQuants {
    float4 *pp; /* particle positions xyzo xyzo ... */
    int n;      /* number of particles              */
};

struct WallTicket {
    RNDunif *rnd;        /* rng on host                                        */
    Clist cells;         /* cell lists (always the same, no need to store map) */
    Texo<int> texstart;  /* texture of starts from clist                       */
    Texo<float4> texpp;  /* texture of particle positions                      */
};
