struct Force;
struct Particle;

// tag::int[]
void scheme_move_apply(float dt, float mass, int n, const Force*, Particle*); // <1>
void scheme_move_clear_vel(int n, Particle*); // <2>
// end::int[]
