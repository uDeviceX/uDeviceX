struct KahanSum;

// tag::interface[]
void kahan_sum_ini(/**/ KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input); // <1>
double kahan_sum_get(const KahanSum*); // <2>
double kahan_sum_compensation(const KahanSum*); // <3>
// end::interface[]
