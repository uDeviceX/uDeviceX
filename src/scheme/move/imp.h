struct MoveParams;
// tag::int[]
void scheme_move_apply(float dt0, MoveParams* moveparams, float mass, int n, const Force*, Particle*); // <1>
void scheme_move_clear_vel(int n, Particle*); // <2>
// end::int[]
