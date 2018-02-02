struct RbcShape;
struct OffRead;
struct Adj;

/* rr: vertices: x0 y0 z0   x1 y1 z1 ... */

// tag::interface[]
void rbc_shape_ini(Adj*, const float *rr, /**/ RbcShape**);
void rbc_shape_fin(/**/ RbcShape*);

void rbc_shape_edg (RbcShape*, /**/ float**);
void rbc_shape_area(RbcShape*, /**/ float**);

void rbc_shape_total_area(RbcShape*, /**/ float*);
void rbc_shape_total_volume(RbcShape*, /**/ float*);
// end::interface[]
