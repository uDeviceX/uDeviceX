void time_ini(float t, /**/ Time** pq) {
    Time *q;
    EMALLOC(1, &q);
    q->curr  = t;
    q->First = 1;
    *pq = q;
}
void time_fin(Time *q) { EFREE(q); }

void time_step(Time *q, float dt) {
    q->First = 0;
    q->prev = q->curr;
    q->curr += dt;
}

float time_current(Time *q) { return q->curr; }

int time_cross(Time *q, float t) {
    float c, p, f;
    if (q->First) return 0;
    if (t <= 0)   ERR("t <= 0");
    c = q->curr; p = q->prev;
    f = floor(c / t) * t;
    return p < f && f < c;
}
