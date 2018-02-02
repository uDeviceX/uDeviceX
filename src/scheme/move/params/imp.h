struct Config;
struct MoveParams;
struct MoveParams_v { };

// tag::mem[]
void scheme_move_params_ini(MoveParams **);
void scheme_move_params_fin(MoveParams *);
// end::mem[]

// tag::cnf[]
void scheme_move_params_conf(const Config *c, MoveParams *);
// end::cnf[]

// tag::int[]
MoveParams_v scheme_move_params_get_view(const MoveParams *);
// end::int[]
