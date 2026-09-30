"""Physical constants used by the pipeline.

All energies are handled in kJ/mol internally (the native unit of GROMACS);
converted to kcal/mol only when writing the final report.
"""

import math

# CODATA exact values
AVOGADRO = 6.02214076e23  # 1/mol
# Boltzmann constant times Avogadro == molar gas constant, in kJ/mol/K.
GAS_CONSTANT = 1.380649e-23 * AVOGADRO * 1.0e-3  # kJ/mol/K

KCAL_PER_KJ = 0.23900574

# Vacuum permittivity in e^2 / (kJ/mol) / nm.
EPS0 = 8.8541878128e-12 / 1.602176634e-19**2 / AVOGADRO / 1e9 * 1e3

# Coulomb constant in kJ/mol * nm / e^2.
KE = 1.0 / (4.0 * math.pi * EPS0)

# Volume of the 1 mol/L standard state, in nm^3.
V0_STANDARD_STATE = 1.6605391


def kbt(temp: float) -> float:
    """RT in kJ/mol for the given temperature in K."""
    return GAS_CONSTANT * temp
