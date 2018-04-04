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

// tag::int[]
void wall_force_apply(const PairParams*, const WvelStep*, const Coords*, const PaArray*, int n, RNDunif*, WallForce,
                      /**/ const FoArray*);
// end::int[]
