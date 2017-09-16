namespace distr {

struct Map {
    int *counts;  /* number of entities leaving in each fragment */
    int *starts;  /* cumulative sum of the above                 */
    int *ids[26]; /* indices of leaving objects                  */
};

} // distr
