#ifndef UDX_EXCH_TYPE_COMMON
#define UDX_EXCH_TYPE_COMMON

#ifndef UDX_MESH_COORD_COMMON
#define UDX_MESH_COORD_COMMON
enum {X, Y, Z};
#endif

template <typename T, int N>
struct Sarray { T d[N]; };

struct Particle { float r[3], v[3]; };
struct Rigid {
    float Iinv[6], mass, com[3], v[3], om[3],
        e0[3], e1[3], e2[3], fo[3], to[3], id;
};
struct Force { float f[3]; };
struct Momentum { float P[3], L[3]; };
struct Positioncp { float rc[3]; float rp[3]; };

typedef Sarray<int, 26> int26;
typedef Sarray<int, 27> int27;
typedef Sarray<int*, 26> intp26;
typedef Sarray<Particle*, 26> Pap26;
typedef Sarray<Force*, 26> Fop26;
typedef Sarray<Momentum*, 26> Mop26;
typedef Sarray<Positioncp*, 26> Pocpp26;

struct PaWrap { int n; const Particle *pp; };
struct FoWrap { int n; Force *ff; };

#endif /* UDX_EXCH_TYPE_COMMON */
