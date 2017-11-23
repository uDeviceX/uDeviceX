struct BulkData {
    float4  *zipped_pp;  /* xyzouvwo xyzouvwo xyzouvwo ...          */
    ushort4 *zipped_rr;  /* xyzo xyzo xyzo...  in half precision    */
    rnd::KISS *rnd;      /* random generator per timestep           */
    const int *colors;   /* pointer to colors, not to be allocated  */
};

struct HaloData {
    flu::LFrag26 lfrags;     /* local fragment infos    */
    flu::RFrag26 rfrags;     /* remote fragment infos   */
    flu::RndFrag26 rndfrags; /* rng and interrank masks */
};
