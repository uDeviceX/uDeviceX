namespace rex {
struct RemoteHalo {
    Particle* dstate;
    Particle* hstate;

    Force* ff;
    Force* ff_pi; /* pinned */
};
}


