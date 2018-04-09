struct VTK;
struct VTKConf;

struct MeshRead;
struct Vectors;
struct Scalars;

// tag::conf[]
void vtk_conf_ini(MeshRead*, /**/ VTKConf**); // <1>
void vtk_conf_fin(VTKConf*);
void vtk_conf_vert(VTKConf*, const char *keys); // <2>
void vtk_conf_tri(VTKConf*, const char *keys); // <3>
// end::conf[]

// tag::main[]
void vtk_ini(MPI_Comm, int maxm, char const *dir, VTKConf*, /**/ VTK**); // <1>
void vtk_points(VTK*, int nm, const Vectors*); // <2>

void vtk_vert(VTK*, int nm, const Scalars*, const char *keys); // <3>
void vtk_tri(VTK*, int nm, const Scalars*, const char *keys);  // <4>

void vtk_write(VTK*, MPI_Comm, int id); // <5>
void vtk_fin(VTK*);
// tag::main[]
