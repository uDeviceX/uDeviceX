struct VTK;
struct VTKConf;

struct MeshRead;

void vtk_conf_ini(MeshRead*, /**/ VTKConf**);
void vtk_conf_fin(VTKConf*);

void vtk_conf_tri(VTKConf*, const char *keys);
