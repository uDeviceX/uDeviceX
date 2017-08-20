namespace rex {
struct RemoteHalo {
    Particle* pp;
    Particle* pp_pi; /* pinned */

    Force* ff;
    Force* ff_pi; /* pinned */
};
}


