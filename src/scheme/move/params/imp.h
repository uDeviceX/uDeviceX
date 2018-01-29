struct Config;
struct MoveParams;
struct MoveParams_v {
    float dt0;
};

void scheme_move_params_ini(MoveParams **);
void scheme_move_params_fin(MoveParams *);

void scheme_move_params_set_timestep(float dt0, MoveParams *);

void scheme_move_params_conf(const Config *c, MoveParams *);

MoveParams_v scheme_move_params_get_view(const MoveParams *);
