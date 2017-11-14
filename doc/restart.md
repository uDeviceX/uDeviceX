# restart

Restart stores simulation variables of solvent, wall, rbcs and rigid bodies

## file format

```
[basedir]/[code]/[XXX].[YYY].[ZZZ]/[ttt].[ext]
```
where:
* `[basedir]` is the base diretory for restart (default: `./strt`
* `[code]` is `flu` (solvent), `wall`, `rbc` or `rig` (rigid bodies)
* `[XXX].[YYY].[ZZZ]` are the coordinates of the processor (no dir if
  single processor)

* `[ttt]` is the id of the restart; it has a magic value `final` for
  the last step of simulation and is used to start from there.
* `[ext]` is the extension of the file: `bop` for particles, `ss` for solids, `id.bop` for particle ids

Special case:
```
[basedir]/[code]/[magic name].[ext]
```
example: template frozen particles from rigid bodies or walls: `templ`

# strt copy

Use `u.strt.cp` to a copy `strt` to a safe place. To run from restart
copy it back to `start` and set `RESTART=true`.

	$ u.start.cp -h

	u.strt.cp [strt dir] [new strt dir]
	copies restart directory
	final.* or [biggest timestep].* becomes final.* in [new strt dir]

`h5` is not recreated after restart use `mkdir -p h5`.
