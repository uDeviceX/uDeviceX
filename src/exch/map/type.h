using namespace comm;

// [E]xchange map
struct EMap {
    int *counts;      /* number of entities leaving in each fragment */
    int *starts;      /* cumulative sum of the above                 */
    int *offsets;     /* offsets per fragment for each solute        */
    int *ids[NFRAGS]; /* indices of leaving objects                  */
};
