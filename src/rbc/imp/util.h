static void diff(const float *a, const float *b, /**/ float *c) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}

static float vabs(float *a) {
    enum {X, Y, Z};
    float r;
    r = a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z];
    return sqrt(r);
}

/* heron formula for triangle area */
static float area_heron(float a, float b, float c) {
  float s;
  s = (a+b+c)/2;
  return sqrt(s*(s-a)*(s-b)*(s-c));
}
