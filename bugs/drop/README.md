# Input

`m`, `kT`, `rc` are 1

	density                                F              6.0
	"water-water" repulsion         alpha_11              12.5
	"οil-oil" repulsion amplitude   alpha_22              12.5
	"οil-water" repulsion amplitude alpha_12              40.0 (See 4.1)
	random force amplitude sigma_11 = sigma_22 = sigma_12 3.35
	time step                                             0.05

# Domain

	10 x 10 x 10

# Drop

Radius is 3, 677 oil particles, 5323 water particles.

# Derived parameters

viscosity

	eta ~ 1.75 L^(-2) (M E)^(1/2) = L^(-2) * M * L/T = M/ (LT)

surface tension

	G   ~ 7.60

# References

* [clark00] Clark, A. T., Lal, M., Ruddock, J. N., & Warren,
  P. B. (2000). Mesoscopic simulation of drops in gravitational and
  shear fields. Langmuir, 16(15), 6342-6350.
