namespace distr {
namespace flu {

/* map helper structure */
struct Map {
    int *counts;  /* number of particles leaving in each fragment */
    int *starts;  /* cumulative sum of the above                  */
    int *ids[26]; /* indices of leaving particles                 */
};

} // flu
} // distr
