namespace rex {
struct LocalHalo {
    int* indexes;

    Force* ff;
    Force* ff_pi; /* pinned */
};

struct RemoteHalo {
    Particle* pp;
    Particle* pp_pi; /* pinned */

    Force* ff;
    Force* ff_pi; /* pinned */
};
}
