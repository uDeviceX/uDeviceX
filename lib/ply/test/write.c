#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "ply.h"
typedef struct Vertex Vertex;
struct Vertex { float x,y,z; };

typedef struct Face Face;
struct Face {
  unsigned char nverts;
  int *verts;
};

PlyProperty vert_props[] = {
  {"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,x), 0, 0, 0, 0},
  {"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,y), 0, 0, 0, 0},
  {"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,z), 0, 0, 0, 0},
};

PlyProperty face_props[] = {
  {"vertex_index", PLY_INT, PLY_INT, offsetof(Face,verts),
   1, PLY_UCHAR, PLY_UCHAR, offsetof(Face,nverts)},
};

int main() {
  int i,j,k;
  PlyFile *ply;
  int nelems;
  char **elist;
  int nprops;
  int num_elems;

  Vertex **vlist;
  Face **flist;
  char *elem_name;

  ply = ply_read(stdin, &nelems, &elist);
  for (i = 0; i < nelems; i++) {
    elem_name = elist[i];
    ply_get_element_description(ply, elem_name, &num_elems, &nprops);
    printf ("element %s %d\n", elem_name, num_elems);
    if (equal_strings ("vertex", elem_name))
      vlist = (Vertex **) malloc (sizeof (Vertex *) * num_elems);
      ply_get_property (ply, elem_name, &vert_props[0]);
      ply_get_property (ply, elem_name, &vert_props[1]);
      ply_get_property (ply, elem_name, &vert_props[2]);
      for (j = 0; j < num_elems; j++) {
        vlist[j] = (Vertex *) malloc (sizeof (Vertex));
        ply_get_element (ply, (void *) vlist[j]);
        printf ("vertex: %g %g %g\n", vlist[j]->x, vlist[j]->y, vlist[j]->z);
      }
    }
    if (equal_strings ("face", elem_name)) {
      flist = (Face **) malloc (sizeof (Face *) * num_elems);
      ply_get_property(ply, elem_name, &face_props[0]);
      for (j = 0; j < num_elems; j++) {
        flist[j] = (Face*) malloc (sizeof (Face));
        ply_get_element (ply, (void *) flist[j]);
        for (k = 0; k < flist[j]->nverts; k++)
          printf ("%d ", flist[j]->verts[k]);
        printf ("\n");
      }
    }
  }
  ply_close (ply);
}
