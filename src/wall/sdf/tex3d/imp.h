struct Tex3d;
struct Tex3d_v;
struct Array3d;

// tag::mem[]
void tex3d_ini(Tex3d**);
void tex3d_fin(Tex3d*);
// end::mem[]

// tag::int[]
void tex3d_copy(const Array3d*, /**/ Tex3d*);    // <1>
void tex3d_to_view(const Tex3d*, /**/ Tex3d_v*); // <2>
// end::int[]
