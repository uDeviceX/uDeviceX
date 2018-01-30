struct FluForcesBulk {
    float4  *zipped_pp;  /* xyzouvwo xyzouvwo xyzouvwo ...          */
    ushort4 *zipped_rr;  /* xyzo xyzo xyzo...  in half precision    */
    RNDunif *rnd;        /* random generator per timestep           */
    const int *colors;   /* pointer to colors, not to be allocated  */

    int3 L; /* subdomain size */
};

struct FluForcesHalo {
    RNDunif *trunks[26];   /* interrank rng (1 per fragment) */
    bool masks[26];        /* interrank masks                */
    
    flu::LFrag26 lfrags;     /* local fragment infos    */
    flu::RFrag26 rfrags;     /* remote fragment infos   */
    flu::RndFrag26 rndfrags; /* rng and interrank masks */

    int3 L; /* subdomain size */
};
