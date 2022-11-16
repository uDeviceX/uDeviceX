// tag::struct[]
/* [D]istr Map */
struct DMap {
    int *counts;  /* number of entities leaving in each fragment */
    int *starts;  /* cumulative sum of the above                 */
    int *ids[27]; /* indices of leaving objects                  */

    int *hcounts; /* counts on host (pinned) */
};
// end::struct[]
