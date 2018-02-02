struct PairParams;

struct WallForce { /* local wall data */
    Sdf_v sdf_v;
    Texo<int> start;
    Texo<float4> pp;
    int n;
    int3 L;
};

void wall_force_apply_color(const PairParams*, Wvel_v wv, const Coords *c, Cloud cloud, int n, RNDunif *rnd, WallForce wa, /**/ Force *ff);
void wall_force_apply(const PairParams*, Wvel_v wv, const Coords *c, Cloud cloud, int n, RNDunif *rnd, WallForce wa, /**/ Force *ff);
