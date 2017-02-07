#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include "rbc_utils.h"

static float  *xx, *yy, *zz;   /* single RBC parameters */
static int   *ff1, *ff2, *ff3;
static long   nb, nf;

static int    nsol;

static float Lx, Ly, Lz; /* simulation domain */
static int   pbcx, pbcy, pbcz;

template <class HDS>
class Build_RBC : public CGAL::Modifier_base<HDS> {
public:
  Build_RBC() {}
  void operator()(HDS& hds) {
    CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);
    B.begin_surface(nb, nf);
    typedef typename HDS::Vertex   Vertex;
    typedef typename Vertex::Point Point;
    for (long iv = 0; iv < nb; iv++) {
      auto x = xx[iv], y = yy[iv], z = zz[iv];
      B.add_vertex(Point(x, y, z));
    }
    for (long ifa = 0; ifa < nf; ifa++) {
      B.begin_facet();
      auto f1 = ff1[ifa], f2 = ff2[ifa], f3 = ff3[ifa];
      B.add_vertex_to_facet(f1);
      B.add_vertex_to_facet(f2);
      B.add_vertex_to_facet(f3);
      B.end_facet();
    }
    B.end_surface();
  }
};

void iotags_init(long nb_, long  nf_, int*   rbc_f1, int*   rbc_f2, int* rbc_f3) {
  nb = nb_;
  nf = nf_; ff1 = rbc_f1; ff2 = rbc_f2; ff3 = rbc_f3; /* set static
							  variables */
}

void iotags_wrap(float* r, float r0, float L) { /* among periodic
					    images of `r' choose the
					    one closest to `r0' */
  auto dr = *r - r0;
  if      (2*dr >  L) *r -= L;
  else if (2*dr < -L) *r += L;
}

namespace ud2f { /* [uD]eviceX rbc definition file [to] [f]aces */
  void ok(ssize_t rc) { /* input file status check */
    if (rc == -1) {
      printf("(ud2faces) ERROR: cannot read uDeviceX file\n");
      exit(EXIT_FAILURE);
    }
  }

  /* nl() realted global variables */
  static char* line; static size_t len; static ssize_t read;
#define nl() ok(read = getline(&line, &len, fd)), line = trim(line) /* [n]ext [l]ine and trim */
  void read_header(FILE* fd) {
    nl(); sscanf(line, "%ld\n", &nb); /* number of vertices in one RBC  */
    nl();                            /* number of edges  */
    nl(); sscanf(line, "%ld\n", &nf);
  }
  bool s_eq(const char* s1, const char* s2) { /* true if `s1' is equal `s2' */
    return std::strcmp(s1, s2) == 0;
  }

  void check_nf(int ifa, int nf) {
    if (ifa != nf) {
      printf("(iotags_init) ERROR: in a header it says: %d faces; I found %d faces\n", nf, ifa);
      exit(EXIT_FAILURE);
    }
  }

  void read_faces(FILE* fd) {
    auto szi = sizeof(int);
    ff1 = (int*)malloc(nf*szi); ff2 = (int*)malloc(nf*szi); ff3 = (int*)malloc(nf*szi);

    do nl(); while (!s_eq(line, "Angles"));
    do nl(); while ( s_eq(line, "")); /* skip empty */
    int iangle, tangle; /* id and type of an `angle'; unused */
    int f1, f2, f3, ifa = 0;
    do {
      sscanf(line, "%d %d %d %d %d\n", &iangle, &tangle, &f1, &f2, &f3);
      ff1[ifa] = f1; ff2[ifa] = f2; ff3[ifa] = f3; ifa++;
      nl();
    } while (read != -1 && !s_eq(line, ""));
    check_nf(ifa, nf);
  }

  void read_file(const char* fn) {
    fprintf(stderr, "(iotags) reading: %s\n", fn);
    auto fd = safe_fopen(fn, "r");
    read_header(fd);
    read_faces(fd);
    fclose(fd);
  }
}

void iotags_init_file(const char* fn) { /* like `iotags_init' but read
					   uDeviceX file */
  ud2f::read_file(fn); /* sets `nb', `nf'; allocates and fills: `ff1',
			  `ff2', `ff3' */
}

void iotags_domain(float Lx_, float Ly_, float Lz_,
		   int pbcx_, int pbcy_, int pbcz_) {
  Lx = Lx_; Ly = Ly_; Lz = Lz_;
  pbcx = pbcx_; pbcy = pbcy_; pbcz = pbcz_;
}

void iotags_single(float* rbc_xx, float* rbc_yy, float* rbc_zz,
		   float* sol_xx, float* sol_yy, float* sol_zz,
		   int* iotags) { /* fills iotags */
  xx = rbc_xx; yy = rbc_yy; zz = rbc_zz; /* set the rest
					    of static variables */
  typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
  typedef CGAL::Polyhedron_3<Kernel>         Polyhedron;
  typedef Polyhedron::HalfedgeDS             HalfedgeDS;
  typedef Kernel::Point_3 Point;

  Build_RBC<HalfedgeDS> RBC {};
  Polyhedron P;
  P.delegate(RBC);
  CGAL::Side_of_triangle_mesh<Polyhedron, Kernel> inside{P};
  for (long isol = 0; isol < nsol; isol++) {
    auto x = sol_xx[isol], y = sol_yy[isol], z = sol_zz[isol];
    
    auto p = Point(x, y, z);
    auto res = inside(p);
    if (res == CGAL::ON_BOUNDED_SIDE) iotags[isol] = 1;
  }
}

/* move all coordinates closer to x0, y0, z0 */
void iotags_recenter(float* xx, float* yy, float* zz,
		     float x0, float y0, float z0) {
  long isol;
  if (pbcx) for (isol = 0; isol < nsol; isol++) iotags_wrap(&xx[isol], x0, Lx);
  if (pbcy) for (isol = 0; isol < nsol; isol++) iotags_wrap(&yy[isol], y0, Ly);
  if (pbcz) for (isol = 0; isol < nsol; isol++) iotags_wrap(&zz[isol], z0, Lz);
}

void iotags_all(long  nrbc , float* rbc_xx, float* rbc_yy, float* rbc_zz,
		long  nsol_, float* sol_xx, float* sol_yy, float* sol_zz,
		int* iotags) { /* fills iotags */
  nsol = nsol_; /* set static */
  auto no = nrbc / nb; /* number of objects (RBCs); nrbc: is the total
			  number of DPD partices belonging to RBCs */
  for (long isol = 0; isol < nsol; isol++) iotags[isol] = 0;
  for (long io = 0; io < no; io++) { /* for every RBC */
    auto x0 = rbc_xx[0], y0 = rbc_yy[0], z0 = rbc_zz[0]; /* any vertex of RBC*/
    iotags_recenter(sol_xx, sol_yy, sol_zz,
    		    x0, y0, z0); /* recenter solvent using periodic BC */
    iotags_single(rbc_xx, rbc_yy, rbc_zz,
  		  sol_xx, sol_yy, sol_zz,
  		  iotags);
    rbc_xx += nb; rbc_yy += nb; rbc_zz += nb;
  }
}
