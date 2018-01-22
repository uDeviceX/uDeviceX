struct WallForce { /* local wall data */
    Sdf_v sdf_v;
    Texo<int> start;
    Texo<float4> pp;
    int n;
};

namespace grey {
void wall_force_apply(Wvel_v wv, Coords c, Cloud cloud, int n, RNDunif *rnd, WallForce wa, /**/ Force *ff);
}

namespace color {
void wall_force_apply(Wvel_v wv, Coords c, Cloud cloud, int n, RNDunif *rnd, WallForce wa, /**/ Force *ff);
}
