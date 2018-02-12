struct Shape0 { float a0, A0, totArea; };

/* extract edge specific shape info */
static void __device__ edg_shape(Shape shape, int, /**/ Shape0 *shape0) {
    shape0->a0      = shape.a0;
    shape0->A0      = shape.A0;
    shape0->totArea = shape.totArea;
}
