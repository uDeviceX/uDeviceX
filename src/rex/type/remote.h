namespace rex {
struct RemoteHalo {
    int n;
    Particle* dstate;
    Particle* hstate;

    Force* ff;
    Force* ff_pi; /* pinned */
    
    Particle* pp;
};
}


