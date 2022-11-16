void kahan_sum_ini(/**/ KahanSum **pq) {
    KahanSum *q;
    EMALLOC(1, &q);
    q->c = q->sum = 0;
    *pq = q;
}

void kahan_sum_fin(KahanSum *q) { EFREE(q); }
void kahan_sum_add(KahanSum *q, double input) {
    double t;
    double y, c, sum;
    y = q->y; c = q->c; sum = q->sum;
    
    y = input - c;
    t = sum + y;
    c = (t - sum) - y;
    sum = t;

    q->y = y; q->c = c; q->sum = sum;
}

double kahan_sum_get(const KahanSum *q) { return q->sum; }
double kahan_sum_compensation(const KahanSum *q) { return q->c; }
