struct Shape0 { /* info for one edge :TODO: */
    float a, b, c;
    float A;
};

static void __device__ edg_shape(int i, Shape shape, /**/ Shape0 *shape0) {
    Edg edg;
    edg = shape.edg[i];
    shape0->a = edg.a;
    shape0->b = edg.b;
    shape0->c = edg.c;
    shape0->A = edg.A;
}
