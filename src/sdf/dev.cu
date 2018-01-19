#include <mpi.h>

#include "math/tform/type.h"
#include "math/tform/imp.h"

#include "tex3d/type.h"
#include "tex3d/imp.h"

#include "imp/type.h"

#include "type.h"
#include "imp.h"

void sdf_to_view(Sdf *q, /**/ Sdf_v *v) {
    tex3d_to_view(q->tex, &v->tex);
    tform_to_view(q->t  , &v->t);
}
