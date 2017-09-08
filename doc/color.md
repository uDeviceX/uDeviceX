# colors

Colors are activated by `#define multi_solvent true`.

Blue: represents outer solvent particles
Red:  represents inner solvent particles

Color is an integer field. It is packed into [Cloud](cloud.md) along
with `pp` arrays. Interactions (fsi, dpdl, dpdr, wall) get
`forces::Pa` from Clouds:

	namespace forces {
	struct Pa {
		float x, y, z, vx, vy, vz;
		int kind;
		int color;
	};

and passes it to `forces::gen`. See
[src/forces/imp.h](src/forces/imp.h) which decides which pair force to
apply based on `int color`. It knows about

	enum {BLUE_COLOR, RED_COLOR};

And miss-uses `gammadpd_solv`, `gammadpd_rbc` and `gammadpd_wall` to
set DPD parameters for blue-blue, red-red, and red-blue interactions.

In `sim::` colors are stored in

	flu::QuantsI     qc;

where

	struct QuantsI {
		int *ii, *ii0;
		int *ii_hst;
	 };

in `sim::sim_gen()` gets colors by calling

	flu::gen_quants(&o::q, &o::qc);`

which has a part different in different [units](u.md). Hard-coded
color schemes are

* `flu/_ussr` : all `BLUE_COLOR` (default)
* `flu/_zurich` : flag of zurich in XY
* `flu/_bangladesh` : a sphere in the center
* `flu/_france` : flag of france in XY

Example [run/color/run](run/color/run)

# recolor

Re-coloring is done every `freq_color` timesteps. `freq_color=0` means
no re-coloring.
