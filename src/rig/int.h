struct Quants {
    int n, ns, nps;
    Particle *pp_hst, *pp;
    Solid *ss_hst, *ss;
    float *rr0_hst, *rr0;
};

struct TicketBB {
    float3 *minbb_hst, *maxbb_hst; /* [b]ounding [b]oxes of solid mesh on host   */
    float3 *minbb_dev, *maxbb_dev; /* [b]ounding [b]oxes of solid mesh on device */

    Solid *ss;
};
