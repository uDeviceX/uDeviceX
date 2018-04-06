struct VTK;
struct VTKConf;

struct MeshRead;
struct Vectors;
struct Scalars;

void vtk_conf_ini(MeshRead*, /**/ VTKConf**);
void vtk_conf_fin(VTKConf*);
void vtk_conf_tri(VTKConf*, const char *keys); /* triangles data */
void vtk_conf_vert(VTKConf*, const char *keys); /* vertices data */

void vtk_ini(MPI_Comm, int maxm, char const *path, VTKConf*, /**/ VTK**);
void vtk_points(VTK*, int nm, const Vectors*);

void vtk_tri(VTK*, int nm, const Scalars*, const char *keys);
void vtk_vert(VTK*, int nm, const Scalars*, const char *keys);

void vtk_write(VTK*, MPI_Comm, int id);
void vtk_fin(VTK*);
