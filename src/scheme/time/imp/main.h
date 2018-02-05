void time_ini(float t, /**/ Time** pq) {
    Time *q;
    EMALLOC(1, &q);
    q->t = q->t0 = t;
    q->First = 1;
    *pq = q;
}
void time_fin(Time *q) { EFREE(q); }

void time_next(Time *q, float dt) {
    if (q->First) {
        q->dt0 = dt;
        q->First = 0;
    } else  {
        q->t0 = q->t;
    }
    q->t  += dt;
    q->dt0 = q->dt;
    q->dt  = dt;
}

float time_current(Time *q) { return q->t; }
float time_dt(Time *q) {
    if (q->First) ERR("time_dt called before time_next");
    return q->dt;
}
float time_dt0(Time *q) { return q->dt0; }
int time_cross(Time *q, float i) {
    float t, t0, f;
    if (q->First) return 0;
    else if (i < 0)   ERR("i < 0");
    else if (i == 0)  return 1;
    
    t = q->t; t0 = q->t0;
    f = floor(t / i) * i;
    return t0 < f && f < t;
}
