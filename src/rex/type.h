namespace rex {
struct LFrag {
    int* indexes;

    Force* ff;
    Force* ff_pi; /* pinned */
};

struct RFrag {
    Particle* pp;
    Particle* pp_pi; /* pinned */

    Force* ff;
    Force* ff_pi; /* pinned */
};
}
