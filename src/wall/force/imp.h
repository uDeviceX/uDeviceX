struct PairParams;
struct PaArray;
struct FoArray;
struct WvelStep;
struct RNDunif;

// tag::struct[]
struct WallForce { /* local wall data */
    Sdf_v sdf_v;
    Texo<int> start;
    Texo<float4> pp;
    int n;
    int3 L;
};
// end::struct[]

struct WallRepulse {
    float l;
};

// tag::int[]
void wall_force_apply(const PairParams*, const WvelStep*, const Coords*, const PaArray*, int n, RNDunif*, WallForce,
                      /**/ const FoArray*); // <1>

void wall_force_adhesion_apply(const PairParams*, const WvelStep*, const Coords*, const PaArray*, int n, RNDunif*, WallForce,
                               /**/ const FoArray*);

void wall_force_repulse(Sdf_v, WallRepulse, long n, const PaArray*, const FoArray*); // <2>
// end::int[]
