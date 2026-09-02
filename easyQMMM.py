import itertools
import os
import shutil
import subprocess
import psi4

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

# 1. USER CONFIGURATION -- edit this section for your system.

PRMTOP_FILE = "system.prmtop"                  # Amber topology, full system: QM + MM atoms
INPCRD_FILE = "system.inpcrd"                  # Amber coordinates (also accepts .rst7)

# 0-based OpenMM atom indices for the QM region, converted from the Amber
# mask '@234-248,1342,1363,1362,1341,1339,1350,1369,1368,1337,1349,1385,1367,1338,1340,1361'
# (Amber atom numbers are 1-based, so each number below is mask_number - 1)
QM_ATOM_INDICES = [
    233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
    1336, 1337, 1338, 1339, 1340, 1341, 1348, 1349, 1360, 1361, 1362, 1366, 1367, 1368, 1384,
]
QM_CHARGE = 1
QM_MULTIPLICITY = 1

# QM Software selection
# "psi4": via the Psi4 Python module.
# "orca": file-based (inpfile.xyz /
#         ptchrg.xyz / orca.inp / orca.engrad), driven exactly the way
#         sander drives ORCA in AmberTools QM/MM.
QM_BACKEND = "orca"                            # "psi4" or "orca"

# Psi4-specific options (used only if QM_BACKEND == "psi4") 
QM_METHOD = "M062X/6-31g*"                     # any Psi4 method/basis string
PSI4_MEMORY = "30 GB"
PSI4_THREADS = 10
# Psi4 SCF/geometry convergence controls, applied globally via
# psi4.set_options() below. Add/remove keys as needed for your method.
PSI4_OPTIONS = {
    "maxiter": 300,          # max SCF iterations
    "guess": "SAD",          # initial guess
    "e_convergence": 1e-6,   # energy convergence
    "d_convergence": 1e-6,   # density convergence
}

# ORCA-specific options (used only if QM_BACKEND == "orca")
ORCA_EXECUTABLE = "/soft/Orca/orca_6_1_0_linux_x86-64_shared_openmpi418/orca"       # FULL path to the orca binary
ORCA_SCRATCH_DIR = "/home/mohamed/MDPHO/QMMMDM/G8/ORCA"              # working directory for
                                                # inpfile.xyz / ptchrg.xyz /
                                                # orca.inp / orca.out /
                                                # orca.engrad -- overwritten
                                                # fresh every QM step

# ORCA INPUT, Modify as you prefer
ORCA_INPUT_HEADER = """\
! M062X 6-31G* NOTRAH RI-SOMF(1X) defgrid1 KDIIS

%scf
SOSCFStart 0.0000033
end

%pal nprocs 6
end
%scf MaxIter 800 end
%maxcore 2000"""

# Embedding scheme (this is the AMBER qmmm_int-style switch)
# "electrostatic": QM Hamiltonian polarizes in the field of the MM point
#                  charges. QM atom charges are zeroed in OpenMM so
#                  there's no double counting of QM-MM electrostatics.
#                  Supported by BOTH backends: Psi4 via its
#                  external_potentials argument, ORCA via its native
#                  %pointcharges "ptchrg.xyz" block.
# "mechanical":    QM sees no external charges at all. QM-MM electrostatics
#                  are handled classically, using the QM region's own
#                  fixed (e.g. RESP) charges already in the prmtop.

EMBEDDING_SCHEME = "electrostatic"             # "electrostatic" or "mechanical"

EMBEDDING_CUTOFF_ANGSTROM = 7.0                # MM charges beyond this are dropped
                                               # (electrostatic scheme only)

# Link atom configuration 
LINK_ATOM_ELEMENT = "H"                        # capping element for every cut bond
LINK_ATOM_SCALE_DEFAULT = 0.723                # r(QM-link) / r(QM-MM), default for a
                                                # C(sp3)-C(sp3) cut capped with H.
                                                # This is NOT universal -- retune per
                                                # bond type (see note below).
# Optional per-bond overrides, keyed by (qm_atom_index, mm_atom_index), for
# cuts where the default C-C scale factor is wrong (e.g. cutting a C-N or
# C-S bond). If a boundary bond isn't listed here, LINK_ATOM_SCALE_DEFAULT
# is used. A reasonable starting value is r_eq(QM-H) / r_eq(QM-MM) using
# typical equilibrium bond lengths for the two bonds involved.

LINK_ATOM_SCALE_OVERRIDES = {
    # (233, 1200): 0.71,
}

# Delete the classical charge of the MM atom directly bonded to the QM
# region (the "M1" atom) from the embedding field, to avoid the QM
# density over-polarizing onto a point charge sitting right at the cut.
# This is the standard "charge deletion" scheme also used by AMBER by
# default; only relevant for EMBEDDING_SCHEME = "electrostatic".
EXCLUDE_BOUNDARY_MM_CHARGE = True

TIMESTEP = 1.0 * unit.femtosecond
TEMPERATURE = 300 * unit.kelvin
FRICTION = 1.0 / unit.picosecond
REPORT_INTERVAL = 1

OUTPUT_INTERVAL_STEPS = REPORT_INTERVAL

QM_XYZ_FILE = "qm_region.xyz"      # snapshot of QM region + link atoms, for checking
SAVE_QM_TRAJECTORY = True

TRAJECTORY_FILE = "trajectory.dcd"
LOG_FILE = "simulation.log"

# 1b. UMBRELLA SAMPLING -- AMBER &rst-style harmonic/flat-bottom restraint
# on a distance or a distance-difference (proton-transfer-style) reaction
# coordinate, biasing the QM/MM dynamics exactly like sander's NMR
# restraints do (r1/r2/r3/r4, rk2/rk3). The restraint is a plain
# classical CustomCompoundBondForce added straight into the OpenMM system
# It works on top of the QM/MM coupling regardless of which
# QM backend or embedding scheme is selected above, and regardless of
# whether the restrained atoms are QM, MM, or one of each.
#
# One run of this script = one umbrella window (like one sander &rst
# job with a fixed r2/r3). To scan windows the way your bash loop over
# $i did, either edit UMBRELLA_R2/UMBRELLA_R3 between runs, or set the
# UMBRELLA_TARGET environment variable per submission, e.g.:
#     for i in $(seq 2.0 0.1 3.0); do
#         UMBRELLA_TARGET=$i python run_qmmm.py
#     done

UMBRELLA_SAMPLING = False                       # master on/off switch

UMBRELLA_MODE = "distance"                     # "distance": r = d(a1,a2)
                                                # "distance_difference": r = d(a1,a2) - d(a3,a4)
                                                #   (e.g. a proton-transfer coordinate:
                                                #   d(donor,H) - d(acceptor,H))
UMBRELLA_ATOMS = (239, 1349)                  # 0-based OpenMM atom indices.
                                                # AMBER's iat= is 1-based -- subtract 1
                                                # from every number when porting an
                                                # existing &rst block, e.g. iat=1195,1031
                                                # -> UMBRELLA_ATOMS = (1194, 1030).
                                                # distance_difference mode needs 4:
                                                # UMBRELLA_ATOMS = (a1, a2, a3, a4)

# Same meaning as AMBER's r1/r2/r3/r4 (Angstrom) and rk2/rk3
# (kcal/mol/Angstrom**2) in a &rst block:
#     r1=0.0, r2=$i, r3=$i, r4=<myvar1>, rk2=200.0, rk3=200.0
# For a single-point umbrella window (the normal case, one target
# value, harmonic on both sides) just set R2 = R3 = your target and
# leave R1/R4 as None; the linear tails then never engage. Give R2 != R3
# only if you deliberately want a flat-bottom restraint over that range.
UMBRELLA_R2 = 2.80                             # target distance, Angstrom (AMBER r2)
UMBRELLA_R3 = 2.80                             # target distance, Angstrom (AMBER r3)
UMBRELLA_R1 = None                             # Angstrom, AMBER r1; None -> 0.0 (never reached by a real distance)
UMBRELLA_R4 = None                             # Angstrom, AMBER r4; None -> 999.0 (never reached)
UMBRELLA_RK2 = 50.0                           # kcal/mol/Angstrom**2, AMBER rk2 (r < r2 side)
UMBRELLA_RK3 = 50.0                           # kcal/mol/Angstrom**2, AMBER rk3 (r > r3 side)

# Optional convenience: override the window target from the shell
# without editing this file, mirroring the AMBER "$i" bash-loop pattern.
_env_umbrella_target = os.environ.get("UMBRELLA_TARGET")
if _env_umbrella_target is not None:
    UMBRELLA_R2 = UMBRELLA_R3 = float(_env_umbrella_target)

UMBRELLA_LABEL = f"win_{UMBRELLA_R2:.3f}"      # tags restart/CV-log filenames for this window
UMBRELLA_CV_LOG_FILE = f"umbrella_cv_{UMBRELLA_LABEL}.dat"   # step, time (ps), r (Angstrom) time series for WHAM
                                                # Logged every OUTPUT_INTERVAL_STEPS MD steps, in lock-step
                                                # with every other output file, set OUTPUT_INTERVAL_STEPS
                                                # to 1 above if you want every step in this file for WHAM.

# 1c. RESTART FILES, write an AMBER-format .rst7 (positions + velocities
# + box) so any stage can be resumed by pointing INPCRD_FILE at it, the
# same as a normal AMBER restart. One is always written at the end of
# every stage; it is ALSO checkpointed periodically (like AMBER's ntwr)
# every OUTPUT_INTERVAL_STEPS steps/outer-iterations, overwriting the
# same file each time, so a killed/crashed run never loses more than one
# interval's worth of progress.

RESTART_FILE_TEMPLATE = "restart_{label}_{stage}.rst7"   # {label} = UMBRELLA_LABEL, {stage} = e.g. "stage2:NVT"

# 1d. SIMULATION PIPELINE, run any sequence of stages, in order. To run
# "MIN only", "EQU (NVT) only", or "PROD (NPT) only", leave every stage
# in the list but zero out the ones you don't want this run: for MIN,
# set max_iterations: 0; for NVT/NPT, set n_steps: 0. A zeroed stage is
# skipped entirely (no minimization call, no MD steps, no reporters) and
# just passes its input coordinates/velocities straight through to the
# next stage.

ENSEMBLE_STAGES = [
    {"ensemble": "MIN", "max_iterations": 20},
    {"ensemble": "NVT", "n_steps": 20},
    {"ensemble": "NPT", "n_steps": 50, "pressure": 1.0 * unit.atmosphere, "barostat_interval": 25},
]


# 1e. QM SETUP -- lazy imports so you only need the package for
# the backend you actually selected.
if QM_BACKEND == "psi4":

    psi4.set_memory(PSI4_MEMORY)
    psi4.set_num_threads(PSI4_THREADS)
    psi4.core.set_output_file("psi4_output.dat", False)
    psi4.set_options(PSI4_OPTIONS)

elif QM_BACKEND == "orca":
    # No Python import needed, ORCA is invoked as an external binary
    # via subprocess for every QM/MM step (see run_orca() below).
    os.makedirs(ORCA_SCRATCH_DIR, exist_ok=True)
    if not os.path.isfile(ORCA_EXECUTABLE) and shutil.which(ORCA_EXECUTABLE) is None:
        print(
            f"WARNING: could not find an ORCA executable at "
            f"'{ORCA_EXECUTABLE}' at startup -- check ORCA_EXECUTABLE "
            f"(use a full path, e.g. '/opt/orca_5_0_4/orca'). The first "
            f"QM step will fail with a clear error if this isn't fixed."
        )

else:
    raise ValueError(f"Unknown QM_BACKEND '{QM_BACKEND}' -- use 'psi4' or 'orca'.")

# Unit conversions
BOHR_PER_ANGSTROM = 1.0 / 0.52917721067
HARTREE_TO_KJMOL = 2625.499639
HARTREE_BOHR_TO_KJMOL_NM = HARTREE_TO_KJMOL / (0.52917721067 * 0.1)
KCAL_TO_KJ = 4.184
AMBER_VELOCITY_UNIT_PS = 20.455    # AMBER internal time-unit constant: velocities in
                                    # a .rst7 are stored in Angstrom per (1/20.455 ps),
                                    # the standard AMBER/ParmEd/cpptraj convention.


# 2. BUILD THE OPENMM SYSTEM

#Every bond crossing the QM/MM boundary (one atom QM, one MM).
def find_boundary_bonds(topology, qm_set):
    boundary_bonds = []
    for bond in topology.bonds():
        a, b = bond.atom1.index, bond.atom2.index
        a_qm, b_qm = a in qm_set, b in qm_set
        if a_qm != b_qm:
            boundary_bonds.append((a, b) if a_qm else (b, a))
    return boundary_bonds


def link_scale_for(qm_i, mm_i):
    return LINK_ATOM_SCALE_OVERRIDES.get((qm_i, mm_i), LINK_ATOM_SCALE_DEFAULT)


def build_system(add_barostat=False, pressure=None, barostat_interval=25):
    prmtop = app.AmberPrmtopFile(PRMTOP_FILE)
    inpcrd = app.AmberInpcrdFile(INPCRD_FILE)

    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )

    if inpcrd.boxVectors is not None:
        system.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)

    qm_set = set(QM_ATOM_INDICES)
    n_atoms = system.getNumParticles()

    symbols = [
        atom.element.symbol if atom.element is not None else "X"
        for atom in prmtop.topology.atoms()
    ]

    # Boundary bonds: identify them, do NOT constrain or freeze them.
    # The MM bond/angle/torsion terms touching them stay exactly as they
    # are in the prmtop; a link atom (built later, per-step) handles the
    # QM side. This is what lets the boundary bond move naturally.
    boundary_bonds = find_boundary_bonds(prmtop.topology, qm_set)
    boundary_mm_atoms = set(mm_i for _, mm_i in boundary_bonds)
    boundary_bond_info = [
        {"qm": qm_i, "mm": mm_i, "scale": link_scale_for(qm_i, mm_i)}
        for qm_i, mm_i in boundary_bonds
    ]

    if boundary_bonds:
        print(f"Found {len(boundary_bonds)} QM/MM boundary bond(s):")
        for info in boundary_bond_info:
            print(f"    QM atom {info['qm']} -- MM atom {info['mm']}  "
                  f"(link scale g = {info['scale']:.3f}, capped with "
                  f"{LINK_ATOM_ELEMENT})")
        print("    -> MM bond/angle/torsion terms across these bonds are left "
              "untouched (not frozen); a link atom caps the QM side each step.")
        if EMBEDDING_SCHEME == "electrostatic" and EXCLUDE_BOUNDARY_MM_CHARGE:
            print(f"    -> excluding {len(boundary_mm_atoms)} directly-bonded MM "
                  f"atom charge(s) from the embedding field (M1 charge deletion).")
    else:
        print("No QM/MM boundary bonds found -- QM_ATOM_INDICES is not "
              "covalently cut, no link atoms needed.")

    def qm_only(indices):
        return all(i in qm_set for i in indices)

    # 2a. Remove bonded terms fully inside the QM region (QM handles these)
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                if qm_only([p1, p2]):
                    force.setBondParameters(i, p1, p2, length, 0.0 * k.unit)
        elif isinstance(force, mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(i)
                if qm_only([p1, p2, p3]):
                    force.setAngleParameters(i, p1, p2, p3, angle, 0.0 * k.unit)
        elif isinstance(force, mm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(i)
                if qm_only([p1, p2, p3, p4]):
                    force.setTorsionParameters(i, p1, p2, p3, p4, per, phase, 0.0 * k.unit)

    # 2b. Nonbonded handling, depends on embedding scheme
    nonbonded = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))

    mm_charges = np.array([
        nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in range(n_atoms)
    ])

    if EMBEDDING_SCHEME == "electrostatic":
        # QM sees MM charges directly -> classical QM charge must be zero,
        # or QM-MM electrostatics would be double counted.
        for i in QM_ATOM_INDICES:
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            nonbonded.setParticleParameters(i, 0.0 * charge.unit, sigma, epsilon)
    elif EMBEDDING_SCHEME == "mechanical":
        # QM sees nothing -> keep the prmtop's QM charges so QM-MM
        # electrostatics are still represented, just classically.
        pass
    else:
        raise ValueError(f"Unknown EMBEDDING_SCHEME '{EMBEDDING_SCHEME}'")

    # Intra-QM electrostatics/LJ must always be zero at the classical level
    # regardless of embedding scheme (QM handles all intra-QM interactions).
    # Existing 1-2/1-3/1-4 exceptions inside the QM region:
    for i in range(nonbonded.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nonbonded.getExceptionParameters(i)
        if qm_only([p1, p2]):
            nonbonded.setExceptionParameters(i, p1, p2, 0.0 * chargeProd.unit, sigma, 0.0 * epsilon.unit)

    # Any QM-QM pair NOT already covered by an exception (e.g. two QM atoms
    # far apart in the topology but both selected into the QM region) would
    # otherwise still interact via the normal nonbonded/PME term. Add an
    # explicit zero exception for every such pair.
    existing_pairs = set()
    for i in range(nonbonded.getNumExceptions()):
        p1, p2, *_ = nonbonded.getExceptionParameters(i)
        existing_pairs.add((min(p1, p2), max(p1, p2)))
    zero_q = 0.0 * unit.elementary_charge ** 2
    zero_eps = 0.0 * unit.kilojoule_per_mole
    one_sigma = 1.0 * unit.nanometer
    for a, b in itertools.combinations(sorted(QM_ATOM_INDICES), 2):
        if (a, b) not in existing_pairs:
            nonbonded.addException(a, b, zero_q, one_sigma, zero_eps, replace=True)

    # QM-MM van der Waals stays classical in both schemes (neither backend
    # is used as a vdW model here), nothing to change there.

    # 2c. CustomExternalForce injects QM (+ redistributed link-atom) forces.
    # Particles include every QM atom AND every boundary MM atom, since a
    # link atom's force gets partly redistributed onto the MM side too.
    qm_force = mm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    qm_force.addPerParticleParameter("fx")
    qm_force.addPerParticleParameter("fy")
    qm_force.addPerParticleParameter("fz")
    qm_force_slot = {}
    for i in list(QM_ATOM_INDICES) + sorted(boundary_mm_atoms):
        if i in qm_force_slot:
            continue
        slot = qm_force.addParticle(i, [0.0, 0.0, 0.0])
        qm_force_slot[i] = slot
    system.addForce(qm_force)

    if add_barostat:
        system.addForce(mm.MonteCarloBarostat(pressure, TEMPERATURE, barostat_interval))

    if UMBRELLA_SAMPLING:
        add_umbrella_restraint(system)

    return (prmtop.topology, inpcrd.positions, system, qm_force, qm_force_slot,
            mm_charges, symbols, boundary_mm_atoms, boundary_bond_info)


# 2b. UMBRELLA SAMPLING RESTRAINT

"""AMBER &rst-style flat-bottom/harmonic restraint:

    r < r1:        rk2*(r1-r2)^2 + 2*rk2*(r1-r2)*(r-r1)   (linear tail)
    r1 <= r < r2:   rk2*(r-r2)^2                            (parabola)
    r2 <= r <= r3:  0                                       (flat)
    r3 < r <= r4:   rk3*(r-r3)^2                            (parabola)
    r > r4:         rk3*(r4-r3)^2 + 2*rk3*(r4-r3)*(r-r4)   (linear tail)

exactly matching sander's NMR restraint potential. Added as a plain
classical CustomCompoundBondForce, so it's integrated by OpenMM like
any other bonded term -- independent of the QM/MM coupling, and
equally valid whether UMBRELLA_ATOMS are QM atoms, MM atoms, or a mix
(e.g. a proton-transfer coordinate straddling the QM/MM boundary).
"""

def add_umbrella_restraint(system):
    r1 = UMBRELLA_R1 if UMBRELLA_R1 is not None else 0.0
    r4 = UMBRELLA_R4 if UMBRELLA_R4 is not None else 999.0
    rk2_kjmol = UMBRELLA_RK2 * KCAL_TO_KJ      # kcal/mol/Angstrom**2 -> kJ/mol/Angstrom**2
    rk3_kjmol = UMBRELLA_RK3 * KCAL_TO_KJ

    if UMBRELLA_MODE == "distance":
        if len(UMBRELLA_ATOMS) != 2:
            raise ValueError("UMBRELLA_MODE = 'distance' needs exactly 2 UMBRELLA_ATOMS")
        n_particles = 2
        r_expr = "10*distance(p1,p2)"          # OpenMM distance() is in nm -> *10 for Angstrom
    elif UMBRELLA_MODE == "distance_difference":
        if len(UMBRELLA_ATOMS) != 4:
            raise ValueError("UMBRELLA_MODE = 'distance_difference' needs exactly 4 UMBRELLA_ATOMS")
        n_particles = 4
        r_expr = "10*(distance(p1,p2) - distance(p3,p4))"
    else:
        raise ValueError(f"Unknown UMBRELLA_MODE '{UMBRELLA_MODE}'")

    energy_expr = (
        "lo_tail*(rk2*(r1-r2)^2 + 2*rk2*(r1-r2)*(r-r1))"
        " + lo_parab*rk2*(r-r2)^2"
        " + hi_parab*rk3*(r-r3)^2"
        " + hi_tail*(rk3*(r4-r3)^2 + 2*rk3*(r4-r3)*(r-r4));"
        "lo_tail = step(r1-r);"
        "lo_parab = step(r2-r)*step(r-r1);"
        "hi_parab = step(r-r3)*step(r4-r);"
        "hi_tail = step(r-r4);"
        f"r = {r_expr}"
    )

    force = mm.CustomCompoundBondForce(n_particles, energy_expr)
    for name, val in [("r1", r1), ("r2", UMBRELLA_R2), ("r3", UMBRELLA_R3), ("r4", r4),
                       ("rk2", rk2_kjmol), ("rk3", rk3_kjmol)]:
        force.addGlobalParameter(name, val)
    force.addBond(list(UMBRELLA_ATOMS), [])
    force.setForceGroup(30)   # kept in its own group so its energy can be isolated if needed
    system.addForce(force)

    return force


#Current value of the umbrella reaction coordinate, in Angstrom
def compute_umbrella_cv(positions_ang):
    if UMBRELLA_MODE == "distance":
        a1, a2 = UMBRELLA_ATOMS
        return float(np.linalg.norm(positions_ang[a1] - positions_ang[a2]))
    else:
        a1, a2, a3, a4 = UMBRELLA_ATOMS
        d12 = np.linalg.norm(positions_ang[a1] - positions_ang[a2])
        d34 = np.linalg.norm(positions_ang[a3] - positions_ang[a4])
        return float(d12 - d34)


# 2c. AMBER-FORMAT RESTART (.rst7) WRITER

#Writes a standard AMBER ASCII restart file
def write_amber_rst7(filename, positions_nm, velocities_nm_ps=None, box_vectors_nm=None, title="restart"):
    positions_ang = np.asarray(positions_nm) * 10.0
    n_atoms = positions_ang.shape[0]

    def _write_block(f, flat_values):
        for i in range(0, len(flat_values), 6):
            f.write("".join(f"{v:12.7f}" for v in flat_values[i:i + 6]) + "\n")

    with open(filename, "w") as f:
        f.write(f"{title}\n")
        f.write(f"{n_atoms:6d}\n")
        _write_block(f, positions_ang.flatten())

        if velocities_nm_ps is not None:
            vel_ang_ps = np.asarray(velocities_nm_ps) * 10.0
            vel_amber = vel_ang_ps / AMBER_VELOCITY_UNIT_PS
            _write_block(f, vel_amber.flatten())

        if box_vectors_nm is not None:
            bv = np.asarray(box_vectors_nm)
            a_len = np.linalg.norm(bv[0]) * 10.0
            b_len = np.linalg.norm(bv[1]) * 10.0
            c_len = np.linalg.norm(bv[2]) * 10.0

            def _angle(u, v):
                cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

            alpha = _angle(bv[1], bv[2])
            beta = _angle(bv[0], bv[2])
            gamma = _angle(bv[0], bv[1])
            f.write(f"{a_len:12.7f}{b_len:12.7f}{c_len:12.7f}{alpha:12.7f}{beta:12.7f}{gamma:12.7f}\n")


def save_restart(simulation, filename, title="restart"):
    state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    velocities_nm_ps = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    box_vectors_nm = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    write_amber_rst7(filename, positions_nm, velocities_nm_ps, box_vectors_nm, title=title)


# 3. LINK ATOMS + QM COUPLING

#R_link = R_qm + g * (R_mm - R_qm), all in Angstrom."""
def link_atom_position(r_qm_ang, r_mm_ang, g):
    return r_qm_ang + g * (r_mm_ang - r_qm_ang)


def build_link_geometry(positions_ang, boundary_bond_info):
    return [
        (info["qm"], info["mm"], info["scale"],
         link_atom_position(positions_ang[info["qm"]], positions_ang[info["mm"]], info["scale"]))
        for info in boundary_bond_info
    ]


"""MM point charges used as the external field for QM. Only ever
called when EMBEDDING_SCHEME == 'electrostatic'.
units="bohr" (Psi4's external_potentials expects Bohr coordinates) or
units="angstrom" (ORCA's ptchrg.xyz format expects Angstrom)."""
def get_embedding_field(positions_ang, qm_set, mm_charges, boundary_mm_atoms=frozenset(),
                         units="bohr"):
    qm_com = positions_ang[list(qm_set)].mean(axis=0)
    coords, charges = [], []
    for i, (pos, q) in enumerate(zip(positions_ang, mm_charges)):
        if i in qm_set or abs(q) < 1e-8:
            continue
        if EXCLUDE_BOUNDARY_MM_CHARGE and i in boundary_mm_atoms:
            continue
        if np.linalg.norm(pos - qm_com) > EMBEDDING_CUTOFF_ANGSTROM:
            continue
        coords.append(pos * BOHR_PER_ANGSTROM if units == "bohr" else pos)
        charges.append(q)
    return coords, charges


def build_qm_cluster_geometry(positions_ang, symbols, boundary_bond_info):
    
    qm_symbols_coords = [
        (symbols[i], positions_ang[i][0], positions_ang[i][1], positions_ang[i][2])
        for i in QM_ATOM_INDICES
    ]
    link_geometry = []  # (qm_idx, mm_idx, g, r_link_ang)
    for info in boundary_bond_info:
        qm_i, mm_i, g = info["qm"], info["mm"], info["scale"]
        r_link = link_atom_position(positions_ang[qm_i], positions_ang[mm_i], g)
        link_geometry.append((qm_i, mm_i, g, r_link))
        qm_symbols_coords.append((LINK_ATOM_ELEMENT, r_link[0], r_link[1], r_link[2]))
    return qm_symbols_coords, link_geometry


def split_link_forces(link_geometry, forces_per_link_atom):
    link_redistrib = []
    for row, (qm_i, mm_i, g, _r_link) in enumerate(link_geometry):
        f_link = forces_per_link_atom[row]
        # R_link = (1-g)*R_qm + g*R_mm
        #   dE/dR_qm gets (1-g) * F_link, dE/dR_mm gets g * F_link
        force_on_qm = (1.0 - g) * f_link
        force_on_mm = g * f_link
        link_redistrib.append((qm_i, mm_i, force_on_qm, force_on_mm))
    return link_redistrib


def build_psi4_molecule(qm_symbols_coords):
    
    lines = [f"{QM_CHARGE} {QM_MULTIPLICITY}"]
    for sym, x, y, z in qm_symbols_coords:
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    lines.append("symmetry c1")
    lines.append("no_reorient")
    lines.append("no_com")
    return psi4.geometry("\n".join(lines))


#Builds the QM+link-atom cluster, runs Psi4
def compute_qm_energy_forces_psi4(positions_nm, symbols, qm_set, mm_charges,
                                   boundary_bond_info):
    positions_ang = positions_nm * 10.0
    qm_symbols_coords, link_geometry = build_qm_cluster_geometry(
        positions_ang, symbols, boundary_bond_info
    )

    mol = build_psi4_molecule(qm_symbols_coords)

    boundary_mm_atoms = frozenset(info["mm"] for info in boundary_bond_info)
    external_potentials = None
    if EMBEDDING_SCHEME == "electrostatic":
        coords_bohr, charges = get_embedding_field(
            positions_ang, qm_set, mm_charges, boundary_mm_atoms, units="bohr"
        )
        if coords_bohr:
            external_potentials = [
                [q, [cx, cy, cz]] for (cx, cy, cz), q in zip(coords_bohr, charges)
            ]

    grad, wfn = psi4.gradient(
        QM_METHOD, molecule=mol, return_wfn=True,
        external_potentials=external_potentials,
    )

    energy_hartree = wfn.energy()
    grad_hartree_bohr = np.array(grad)
    forces_hartree_bohr = -grad_hartree_bohr  # force = -gradient

    forces_kjmol_nm = forces_hartree_bohr * HARTREE_BOHR_TO_KJMOL_NM
    energy_kjmol = energy_hartree * HARTREE_TO_KJMOL

    n_real = len(QM_ATOM_INDICES)
    direct_forces = {
        QM_ATOM_INDICES[k]: forces_kjmol_nm[k] for k in range(n_real)
    }
    link_redistrib = split_link_forces(link_geometry, forces_kjmol_nm[n_real:])

    return energy_kjmol, direct_forces, link_redistrib


# File-based, out-of-process, following the same layout ORCA is normally

_ORCA_XYZ = "inpfile.xyz"
_ORCA_PTCHRG = "ptchrg.xyz"
_ORCA_INP = "orca.inp"
_ORCA_OUT = "orca.out"
_ORCA_ENGRAD = "orca.engrad"   # ORCA names <basename>.engrad from <basename>.inp


def write_orca_xyz(filename, qm_symbols_coords):
    with open(filename, "w") as f:
        f.write(f"{len(qm_symbols_coords)}\n")
        f.write("QM/MM cluster geometry (real QM atoms + link atoms), Angstrom\n")
        for sym, x, y, z in qm_symbols_coords:
            f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")


#ORCA's %pointcharges file format

def write_orca_ptcharges(filename, coords_ang, charges):
    with open(filename, "w") as f:
        f.write(f"{len(charges)}\n")
        for (x, y, z), q in zip(coords_ang, charges):
            f.write(f"{q:14.8f} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def write_orca_input(filename, use_pointcharges):
    lines = [ORCA_INPUT_HEADER.rstrip("\n"), "! ENGRAD", "! Angs NoUseSym"]
    if use_pointcharges:
        lines.append(f'%pointcharges "{_ORCA_PTCHRG}"')
    lines.append(f"*xyzfile {QM_CHARGE} {QM_MULTIPLICITY} {_ORCA_XYZ}")
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_orca(cwd):
    with open(os.path.join(cwd, _ORCA_OUT), "w") as out:
        result = subprocess.run(
            [ORCA_EXECUTABLE, _ORCA_INP],
            stdout=out, stderr=subprocess.STDOUT, cwd=cwd,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"ORCA exited with code {result.returncode} -- check "
            f"{os.path.join(cwd, _ORCA_OUT)} for details."
        )


def parse_orca_engrad(filename, n_atoms_expected):
    
    with open(filename) as f:
        raw_lines = [ln.strip() for ln in f]
    data_lines = [ln for ln in raw_lines if ln and not ln.startswith("#")]

    idx = 0
    n_atoms_file = int(data_lines[idx]); idx += 1
    if n_atoms_file != n_atoms_expected:
        raise ValueError(
            f"ORCA .engrad reports {n_atoms_file} atoms but the QM+link "
            f"cluster sent had {n_atoms_expected} -- geometry/engrad "
            f"mismatch, don't trust this step."
        )

    energy_hartree = float(data_lines[idx]); idx += 1

    gradient = np.zeros((n_atoms_expected, 3))
    flat = data_lines[idx: idx + 3 * n_atoms_expected]
    for i in range(n_atoms_expected):
        gradient[i, 0] = float(flat[3 * i])
        gradient[i, 1] = float(flat[3 * i + 1])
        gradient[i, 2] = float(flat[3 * i + 2])

    return energy_hartree, gradient


def compute_qm_energy_forces_orca(positions_nm, symbols, qm_set, mm_charges,
                                   boundary_bond_info):
    positions_ang = positions_nm * 10.0
    qm_symbols_coords, link_geometry = build_qm_cluster_geometry(
        positions_ang, symbols, boundary_bond_info
    )

    boundary_mm_atoms = frozenset(info["mm"] for info in boundary_bond_info)
    use_pointcharges = (EMBEDDING_SCHEME == "electrostatic")

    write_orca_xyz(os.path.join(ORCA_SCRATCH_DIR, _ORCA_XYZ), qm_symbols_coords)

    if use_pointcharges:
        coords_ang, charges = get_embedding_field(
            positions_ang, qm_set, mm_charges, boundary_mm_atoms, units="angstrom"
        )
        write_orca_ptcharges(os.path.join(ORCA_SCRATCH_DIR, _ORCA_PTCHRG), coords_ang, charges)

    write_orca_input(os.path.join(ORCA_SCRATCH_DIR, _ORCA_INP), use_pointcharges)

    run_orca(ORCA_SCRATCH_DIR)

    n_cluster = len(qm_symbols_coords)
    energy_hartree, gradient_hartree_bohr = parse_orca_engrad(
        os.path.join(ORCA_SCRATCH_DIR, _ORCA_ENGRAD), n_cluster
    )
    forces_hartree_bohr = -gradient_hartree_bohr  # force = -gradient

    forces_kjmol_nm = forces_hartree_bohr * HARTREE_BOHR_TO_KJMOL_NM
    energy_kjmol = energy_hartree * HARTREE_TO_KJMOL

    n_real = len(QM_ATOM_INDICES)
    direct_forces = {
        QM_ATOM_INDICES[k]: forces_kjmol_nm[k] for k in range(n_real)
    }
    link_redistrib = split_link_forces(link_geometry, forces_kjmol_nm[n_real:])

    return energy_kjmol, direct_forces, link_redistrib


def compute_qm_energy_forces(positions_nm, symbols, qm_set, mm_charges,
                              boundary_bond_info):
    if QM_BACKEND == "psi4":
        return compute_qm_energy_forces_psi4(
            positions_nm, symbols, qm_set, mm_charges, boundary_bond_info
        )
    elif QM_BACKEND == "orca":
        return compute_qm_energy_forces_orca(
            positions_nm, symbols, qm_set, mm_charges, boundary_bond_info
        )
    else:
        raise ValueError(f"Unknown QM_BACKEND '{QM_BACKEND}'")


def write_qm_xyz(filename, symbols, positions_ang, qm_indices, link_geometry=None,
                  comment="", mode="w"):
    n_link = len(link_geometry) if link_geometry else 0
    with open(filename, mode) as f:
        f.write(f"{len(qm_indices) + n_link}\n")
        f.write(f"{comment}\n")
        for i in qm_indices:
            x, y, z = positions_ang[i]
            f.write(f"{symbols[i]:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")
        if link_geometry:
            for _qm_i, _mm_i, _g, r_link in link_geometry:
                f.write(f"{LINK_ATOM_ELEMENT:2s} {r_link[0]:14.8f} {r_link[1]:14.8f} {r_link[2]:14.8f}\n")


# 4. MAIN MD LOOP

def make_simulation(topology, system, positions, velocities=None, box_vectors=None):
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)

    simulation = None
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
        # Merely finding the CUDA platform doesn't guarantee it can
        # actually create a Context on this GPU/driver -- a PTX/driver
        # version mismatch (CUDA_ERROR_UNSUPPORTED_PTX_VERSION) only
        # surfaces here, at Context creation, not at getPlatformByName().
        simulation = app.Simulation(topology, system, integrator, platform)
    except Exception as exc:
        print(f"CUDA platform unavailable or failed to initialize ({exc}); "
              f"falling back to CPU. If this is unexpected, check "
              f"`nvidia-smi` (driver's max CUDA version) against the "
              f"cudatoolkit/cuda-version OpenMM was built against.")
        platform = mm.Platform.getPlatformByName("CPU")
        integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)
        simulation = app.Simulation(topology, system, integrator, platform)

    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)

    simulation.context.setPositions(positions)

    if velocities is not None:
        simulation.context.setVelocities(velocities)
    else:
        simulation.context.setVelocitiesToTemperature(TEMPERATURE)

    return simulation


def update_qm_force(simulation, qm_force, qm_force_slot, symbols, qm_set,
                      mm_charges, boundary_bond_info):
    state = simulation.context.getState(getPositions=True)
    positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    qm_energy, direct_forces, link_redistrib = compute_qm_energy_forces(
        positions_nm, symbols, qm_set, mm_charges, boundary_bond_info
    )

    totals = {i: np.zeros(3) for i in qm_force_slot}
    for i, f in direct_forces.items():
        totals[i] += f
    for qm_i, mm_i, f_qm, f_mm in link_redistrib:
        totals[qm_i] += f_qm
        totals[mm_i] += f_mm

    for i, slot in qm_force_slot.items():
        fx, fy, fz = totals[i]
        qm_force.setParticleParameters(slot, i, [fx, fy, fz])
    qm_force.updateParametersInContext(simulation.context)

    return qm_energy


def emit_outputs(simulation, symbols, boundary_bond_info, label, step_index,
                   qm_energy, restart_path=None, time_ps=None):
    
    state = simulation.context.getState(getPositions=True)
    positions_ang = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    if UMBRELLA_SAMPLING:
        r_cv = compute_umbrella_cv(positions_ang)
        t = time_ps if time_ps is not None else 0.0
        with open(UMBRELLA_CV_LOG_FILE, "a") as f:
            f.write(f"{label:>14s} {step_index:8d} {t:12.5f} {r_cv:12.5f}\n")

    if SAVE_QM_TRAJECTORY:
        link_geometry = build_link_geometry(positions_ang, boundary_bond_info)
        write_qm_xyz(
            QM_XYZ_FILE, symbols, positions_ang, QM_ATOM_INDICES, link_geometry,
            comment=f"{label} step {step_index}, QM E = {qm_energy:.6f} kJ/mol", mode="a",
        )

    if restart_path:
        save_restart(simulation, restart_path, title=f"{label} step {step_index}")


def run_qmmm_minimization(simulation, qm_force, qm_force_slot, symbols, qm_set,
                           mm_charges, boundary_bond_info, label,
                           outer_iterations=40, inner_mm_iterations=5,
                           restart_path=None):
    prev_energy = None
    for outer in range(outer_iterations):
        qm_energy = update_qm_force(
            simulation, qm_force, qm_force_slot, symbols, qm_set,
            mm_charges, boundary_bond_info,
        )
        simulation.minimizeEnergy(maxIterations=inner_mm_iterations)

        mm_state = simulation.context.getState(getEnergy=True)
        mm_energy = mm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f"[{label}] outer iter {outer:4d}  QM E = {qm_energy:12.4f} kJ/mol  "
              f"MM(+injected) E = {mm_energy:12.4f} kJ/mol")

        if outer % OUTPUT_INTERVAL_STEPS == 0:
            emit_outputs(
                simulation, symbols, boundary_bond_info, label, outer,
                qm_energy, restart_path=restart_path, time_ps=None,
            )

        if prev_energy is not None and abs(mm_energy - prev_energy) < 1.0:
            print(f"[{label}] converged (\u0394E < 1 kJ/mol) after {outer + 1} outer iterations")
            break
        prev_energy = mm_energy

    # One last QM evaluation at the final geometry, so the force handed
    # off to the next stage (NVT/NPT) is consistent with where atoms
    # actually ended up, not one outer-iteration stale -- and a final
    # output emission so qm_region.xyz / the CV log / the restart file
    # all reflect that final geometry too.
    qm_energy = update_qm_force(
        simulation, qm_force, qm_force_slot, symbols, qm_set,
        mm_charges, boundary_bond_info,
    )
    emit_outputs(
        simulation, symbols, boundary_bond_info, label, outer,
        qm_energy, restart_path=restart_path, time_ps=None,
    )


def run_qmmm_md(simulation, qm_force, qm_force_slot, symbols, qm_set, mm_charges,
                 n_steps, label, boundary_bond_info, restart_path=None):
    for step in range(n_steps):
        qm_energy = update_qm_force(
            simulation, qm_force, qm_force_slot, symbols, qm_set,
            mm_charges, boundary_bond_info,
        )

        simulation.step(1)

        need_output = (step % OUTPUT_INTERVAL_STEPS == 0) or (step == n_steps - 1)

        if need_output:
            mm_state = simulation.context.getState(getEnergy=True)
            mm_energy = mm_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            print(f"[{label}] Step {step:6d}  QM E = {qm_energy:12.4f} kJ/mol  "
                  f"MM(+injected) E = {mm_energy:12.4f} kJ/mol")

            time_ps = (step + 1) * TIMESTEP.value_in_unit(unit.picosecond)
            emit_outputs(
                simulation, symbols, boundary_bond_info, label, step,
                qm_energy, restart_path=restart_path, time_ps=time_ps,
            )


def main():
    inpcrd = app.AmberInpcrdFile(INPCRD_FILE)
    positions = inpcrd.positions
    velocities = None
    box_vectors = inpcrd.boxVectors

    qm_set = set(QM_ATOM_INDICES)
    wrote_initial_snapshot = False
    dcd_started = False

    if UMBRELLA_SAMPLING:
        cv_desc = (f"d({UMBRELLA_ATOMS[0]},{UMBRELLA_ATOMS[1]})" if UMBRELLA_MODE == "distance"
                   else f"d({UMBRELLA_ATOMS[0]},{UMBRELLA_ATOMS[1]}) - d({UMBRELLA_ATOMS[2]},{UMBRELLA_ATOMS[3]})")
        with open(UMBRELLA_CV_LOG_FILE, "w") as f:
            f.write(f"# umbrella window: r2={UMBRELLA_R2} r3={UMBRELLA_R3} "
                    f"rk2={UMBRELLA_RK2} rk3={UMBRELLA_RK3} kcal/mol/Angstrom**2\n")
            f.write(f"# CV = {cv_desc}, logged every {OUTPUT_INTERVAL_STEPS} step(s)\n")
            f.write(f"# {'step':>6s} {'time_ps':>12s} {'r_Angstrom':>12s}\n")

    for stage_index, stage in enumerate(ENSEMBLE_STAGES):
        ensemble = stage["ensemble"].upper()
        label = f"stage{stage_index + 1}:{ensemble}"

        if ensemble in ("NVT", "NPT") and stage.get("n_steps", 0) == 0:
            print(f"\n=== {label} === (skipped: n_steps = 0)")
            continue
        if ensemble == "MIN" and stage.get("max_iterations", 0) == 0:
            print(f"\n=== {label} === (skipped: max_iterations = 0)")
            continue

        print(f"\n=== {label} ===")
        label_fs = label.replace(":", "_")
        
        restart_path = RESTART_FILE_TEMPLATE.format(label=UMBRELLA_LABEL, stage=label_fs) \
            if UMBRELLA_SAMPLING else RESTART_FILE_TEMPLATE.format(label="nowin", stage=label_fs)
        add_barostat = (ensemble == "NPT")
        system_kwargs = {}
        if add_barostat:
            system_kwargs["pressure"] = stage.get("pressure", 1.0 * unit.atmosphere)
            system_kwargs["barostat_interval"] = stage.get("barostat_interval", 25)

        (topology, _default_positions, system, qm_force, qm_force_slot,
         mm_charges, symbols, boundary_mm_atoms, boundary_bond_info) = \
            build_system(add_barostat=add_barostat, **system_kwargs)

        simulation = make_simulation(topology, system, positions, velocities, box_vectors)

        if not wrote_initial_snapshot:
            positions_ang = np.array(
                simulation.context.getState(getPositions=True)
                .getPositions(asNumpy=True)
                .value_in_unit(unit.angstrom)
            )
            link_geometry = build_link_geometry(positions_ang, boundary_bond_info)
            write_qm_xyz(
                QM_XYZ_FILE, symbols, positions_ang, QM_ATOM_INDICES, link_geometry,
                comment="QM region + link atoms, as loaded (before first stage)", mode="w",
            )
            print(f"Wrote initial QM region geometry to {QM_XYZ_FILE} "
                  f"({len(QM_ATOM_INDICES)} QM atoms + {len(link_geometry)} link atom(s)) "
                  f"-- open it to check the selection.")
            wrote_initial_snapshot = True

        if ensemble == "MIN":
            run_qmmm_minimization(
                simulation, qm_force, qm_force_slot, symbols, qm_set, mm_charges,
                boundary_bond_info, label=label,
                outer_iterations=stage.get("max_iterations", 200) // 5 or 1,
                inner_mm_iterations=5,
                restart_path=restart_path,
            )
        elif ensemble in ("NVT", "NPT"):
            simulation.reporters.append(
                app.DCDReporter(TRAJECTORY_FILE, OUTPUT_INTERVAL_STEPS, append=dcd_started)
            )
            simulation.reporters.append(app.StateDataReporter(
                LOG_FILE, OUTPUT_INTERVAL_STEPS, step=True, time=True,
                potentialEnergy=True, kineticEnergy=True, temperature=True,
                volume=add_barostat, speed=True, append=dcd_started,
            ))
            dcd_started = True

            run_qmmm_md(
                simulation, qm_force, qm_force_slot, symbols, qm_set, mm_charges,
                n_steps=stage["n_steps"], label=label, boundary_bond_info=boundary_bond_info,
                restart_path=restart_path,
            )
        else:
            raise ValueError(f"Unknown ensemble '{stage['ensemble']}' in ENSEMBLE_STAGES")

        final_state = simulation.context.getState(
            getPositions=True, getVelocities=True, enforcePeriodicBox=True
        )
        positions = final_state.getPositions()
        velocities = final_state.getVelocities()
        box_vectors = final_state.getPeriodicBoxVectors()

        save_restart(simulation, restart_path, title=f"{label} final")

    print("\nAll stages complete.")

if __name__ == "__main__":
    main()
