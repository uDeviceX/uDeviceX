// tag::interface[]
void ini(WvelCste p, Wvel *vw);
void ini(WvelShear p, Wvel *vw);
void ini(WvelShearSin p, Wvel *vw);
void ini(WvelHS p, Wvel *vw);

void step2params(long it, const Wvel *wv, /**/ Wvel_v *view);
// end::interface[]
