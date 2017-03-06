namespace field {
  float *data,  extent[3];
  int N[3];

  template <int k> struct Bspline {
  template <int i> static float eval(float x) {
    return (x - i) / (k - 1) * Bspline<k - 1>::template eval<i>(x) +
      (i + k - x) / (k - 1) * Bspline<k - 1>::template eval<i + 1>(x);
  }
  };

  template <> struct Bspline<1> {
    template <int i> static float eval(float x) {
      return (float)(i) <= x && x < (float)(i + 1);
    }
  };
}
