namespace rex {
struct RemoteHalo {
    Particle* pp;
    Particle* hstate;

    Force* ff;
    Force* ff_pi; /* pinned */
};
}


