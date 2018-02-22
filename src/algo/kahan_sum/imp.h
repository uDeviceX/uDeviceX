struct KahanSum;

void kahan_sum_ini(/**/ KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input);
double kahan_sum_get(const KahanSum*);
double kahan_sum_compensation(const KahanSum*);
