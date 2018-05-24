// tag::int[]
void emap_ini(int nw, int nfrags, int cap[], /**/ EMap *map);               // <1>
void emap_fin(int nfrags, EMap *map);                                       // <2>
void emap_reini(int nw, int nfrags, /**/ EMap map);                         // <3>
void emap_scan(int nw, int nfrags, /**/ EMap map);                          // <4>
void emap_download_tot_counts(int nw, int nfrags, EMap map, /**/ int counts[]); // <5>
void emap_download_all_counts(int nw, int nfrags, EMap map, /**/ int tot_counts[], int *counts[]);
// end::int[]
