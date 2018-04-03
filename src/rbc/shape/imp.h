struct RbcShape;
struct MeshRead;
struct Adj;

/* rr: vertices: x0 y0 z0   x1 y1 z1 ... */

// tag::mem[]
void rbc_shape_ini(const Adj*, const float *rr, /**/ RbcShape**);
void rbc_shape_fin(/**/ RbcShape*);
// end::mem[]

// tag::int[]
void rbc_shape_edg (RbcShape*, /**/ float**); // <1>
void rbc_shape_area(RbcShape*, /**/ float**); // <2>

void rbc_shape_total_area(RbcShape*, /**/ float*);    // <3>
void rbc_shape_total_volume(RbcShape*, /**/ float*);  // <4>
// end::int[]
