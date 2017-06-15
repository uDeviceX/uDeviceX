#define cross(a, b) make_float3                 \
    ((a).y*(b).z - (a).z*(b).y,                 \
     (a).z*(b).x - (a).x*(b).z,                 \
     (a).x*(b).y - (a).y*(b).x)
