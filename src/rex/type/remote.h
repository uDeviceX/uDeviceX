namespace rex {
struct RemoteHalo {
    Particle* dstate;
    Particle* hstate;
    PinnedHostBuffer<Force> ff;
    Particle* pp;
    int n;
};

namespace re {
void resize(RemoteHalo *r, int n) {
    r->n = n;
    r->ff.resize(n);
}
}
}
