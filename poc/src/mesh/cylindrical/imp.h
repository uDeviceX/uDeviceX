struct MeshCylindrical;
struct Vectors;

// tag::interface[]
void mesh_cylindrical_ini(int nv, /**/ MeshCylindrical**);
void mesh_cylindrical_fin(MeshCylindrical*);
void  mesh_cylindrical_apply(MeshCylindrical*, int nm, Vectors*, /**/
                           double *r, double *phi, double *z);
// end::interface[]
