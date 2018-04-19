struct Objects;
struct Config;

void objects_ini(const Config*, Objects**);
void objects_fin(Objects*);

void objects_update(Objects*);
void objects_distribute(Objects*);
void objects_dump(Objects*);

