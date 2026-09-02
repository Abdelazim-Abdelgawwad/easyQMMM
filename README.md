# easyQMMM

**easyQMMM** is a single-file, AMBER-style **additive QM/MM** molecular dynamics
driver that couples [OpenMM](https://openmm.org/) (classical MM region) with a
choice of QM engine (**Psi4** or **ORCA**) for the QM region. It follows the same
QM/MM scheme used by `sander`/`pmemd` in AmberTools (Walker, Crowley & Case,
*J. Comput. Chem.* 2008), and supports both plain (**unbiased**) QM/MM dynamics and
**umbrella-sampling (biased) QM/MM dynamics** along a reaction coordinate — all
from a single configuration section, no separate QM/MM interface software needed.

easyQMMM is designed as a companion to [easyPARM](https://github.com/<your-username>/easyPARM):
easyPARM builds the force field parameters (including for metal-containing
systems), and easyQMMM runs the QM/MM dynamics on top of them.

---

## Features

- **Two QM backends, one interface**
  - `psi4` — in-process, via the Psi4 Python module (ab initio / DFT).
  - `orca` — out-of-process, file-based (`inpfile.xyz` / `ptchrg.xyz` / `orca.inp` /
    `orca.engrad`), driven exactly the way `sander` drives ORCA in AmberTools QM/MM.
- **Two embedding schemes**
  - `mechanical` — QM sees no MM charges; QM–MM electrostatics come from the QM
    region's own fixed force-field (e.g. RESP) charges (like `qmmm_int = 1`).
  - `electrostatic` — QM is polarized directly by the surrounding MM point charges
    (like `qmmm_int = 5` / Gaussian ONIOM electronic embedding), with optional M1
    charge deletion at the boundary and a distance cutoff for included MM charges.
- **Genuine link atoms** at every QM/MM boundary bond (not a frozen boundary bond),
  with per-bond, tunable scale factors (`r(QM–link) / r(QM–MM)`).
- **Unbiased QM/MM MD** — standard MIN → NVT → NPT staged dynamics.
- **Biased QM/MM MD (umbrella sampling)** — an AMBER `&rst`-style
  harmonic/flat-bottom restraint (`r1/r2/r3/r4`, `rk2/rk3`) on a distance or a
  distance-difference (proton-transfer-style) reaction coordinate, for building
  PMFs with WHAM/MBAR. One run = one umbrella window, with a CV time-series log
  written per window.
- **AMBER-format restart files (`.rst7`)** written after every stage and
  periodically during long stages, so any stage can be resumed the same way an
  AMBER restart is resumed.
- **Configurable staged pipeline** (`ENSEMBLE_STAGES`) — run any sequence of
  MIN / NVT / NPT stages, or skip a stage entirely by zeroing its steps/iterations.
- Automatic **CUDA → CPU fallback** if no compatible GPU platform is found.

---

## How it works (brief)

1. The full system (QM + MM, already solvated/prepared, e.g. via `tleap`) is loaded
   from an Amber `prmtop`/`inpcrd`.
2. Bonded MM terms fully inside the QM region are zeroed out (QM handles those).
3. Depending on the embedding scheme, QM atom charges in the MM `NonbondedForce`
   are zeroed (electrostatic embedding) or left alone (mechanical embedding).
   Intra-QM nonbonded interactions are always zeroed classically.
4. Every QM/MM step: a QM+link-atom cluster geometry is built, sent to Psi4 or
   ORCA (with or without an MM point-charge field), and the returned energy/
   gradient is converted to forces, redistributed across real QM atoms and the
   MM atoms bonded across the boundary, and injected into OpenMM via a
   `CustomExternalForce`.
5. OpenMM integrates the combined MM + injected-QM forces one step (or one
   minimization sub-cycle) at a time.
6. If umbrella sampling is enabled, a classical `CustomCompoundBondForce`
   restraint biases a chosen distance or distance-difference coordinate, and its
   value is logged every `OUTPUT_INTERVAL_STEPS` steps for later PMF reconstruction.

---

## Requirements / Dependencies

| Component | Needed for | Notes |
|---|---|---|
| Python ≥ 3.9 | everything | |
| [OpenMM](https://openmm.org/) | everything | MM engine + integrator |
| NumPy | everything | |
| [Psi4](https://psicode.org/) | `QM_BACKEND = "psi4"` only | not available via pip; conda only |
| [ORCA](https://orcaforum.kofo.mpg.de/) ≥ 5.x | `QM_BACKEND = "orca"` only | separate license required; not installed via pip/conda |
| AmberTools (`tleap`, etc.) | system prep | to build the `prmtop`/`inpcrd`, not required at runtime |
| CUDA-capable GPU + driver (optional) | speed | script auto-falls back to CPU if unavailable |

---

## Installation

It's strongly recommended to use a dedicated conda environment.

```bash
# 1. Create and activate an environment
conda create -n qmmm python=3.10 -y
conda activate qmmm

# 2. OpenMM + NumPy
conda install -c conda-forge openmm numpy -y
# (equivalently: pip install openmm numpy)

# 3a. Psi4 (only if you plan to use QM_BACKEND = "psi4")
conda install -c psi4/label/dev -c conda-forge psi4 -y

# 3b. ORCA (only if you plan to use QM_BACKEND = "orca")
#   ORCA is not distributed via pip/conda. Download it from the official
#   ORCA forum (license required) and install it yourself, e.g. under
#   /opt/orca_6_1_0_linux_x86-64_shared_openmpi418/
#   Make sure any MPI version ORCA needs (e.g. OpenMPI) is on your PATH.
```

Verify the install:

```bash
python -c "import openmm; print(openmm.version.version)"
python -c "import psi4; print(psi4.__version__)"        # if using Psi4
/opt/orca_6_1_0_linux_x86-64_shared_openmpi418/orca --help  # if using ORCA
```

Clone this repository:

```bash
git clone https://github.com/<your-username>/easyQMMM.git
cd easyQMMM
```

---

## Example system

An example `system.prmtop` / `system.inpcrd` pair is included in this
repository so you can run easyQMMM out of the box without first building your
own topology. This example reproduces the **barrierless nucleophilic attack of
a reactive carbocation on a DNA base**, exactly as characterized by hybrid
QM/MM dynamics in our published work on photoinduced DNA interstrand
cross-linking:

> Abdelgawwad, A. M. A.; Monari, A.; Tuñón, I.; Francés-Monerris, A. *Spatial
> and Temporal Resolution of the Oxygen-Independent Photoinduced DNA
> Interstrand Cross-Linking by a Nitroimidazole Derivative.* **J. Chem. Inf.
> Model. 2022**, *62*, 3239–3252.
> [DOI: 10.1021/acs.jcim.2c00460](https://doi.org/10.1021/acs.jcim.2c00460)

In that study, the QM/MM simulations showed that the carbocation intermediate
attacks the most nucleophilic DNA positions — the N7 and O6 sites of guanine,
the N3 and N7 sites of adenine, the N4 site of cytosine, and the O2 site of
thymine — through essentially barrierless reaction pathways on the picosecond
time scale. The bundled example system is set up so that running easyQMMM
on it reproduces this behavior directly, giving you a ready-made,
literature-validated test case to confirm your installation is working
correctly before moving on to your own systems.

---

## Usage

### 1. Prepare your system

Build a fully solvated/parameterized Amber topology and coordinates as usual
(`tleap` → `system.prmtop`, `system.inpcrd`).

### 2. Configure the script

Open the script and edit the **USER CONFIGURATION** section at the top:

- `PRMTOP_FILE`, `INPCRD_FILE` — paths to your system.
- `QM_ATOM_INDICES` — 0-based OpenMM atom indices for the QM region (subtract 1
  from any 1-based Amber mask atom numbers).
- `QM_CHARGE`, `QM_MULTIPLICITY`.
- `QM_BACKEND` — `"psi4"` or `"orca"`.
  - Psi4: set `QM_METHOD`, `PSI4_MEMORY`, `PSI4_THREADS`, `PSI4_OPTIONS`.
  - ORCA: set `ORCA_EXECUTABLE` (full path, not just `"orca"`), `ORCA_SCRATCH_DIR`,
    and `ORCA_INPUT_HEADER` (functional, basis, ECPs, `%pal`, `%maxcore`, etc.).
- `EMBEDDING_SCHEME` — `"electrostatic"` or `"mechanical"`, and
  `EMBEDDING_CUTOFF_ANGSTROM` for the electrostatic MM-charge cutoff.
- `LINK_ATOM_SCALE_DEFAULT` / `LINK_ATOM_SCALE_OVERRIDES` — tune per boundary
  bond type; the default (0.723) is for a C(sp3)–C(sp3) cut capped with H.
- `ENSEMBLE_STAGES` — the MIN/NVT/NPT pipeline to run (zero out steps/iterations
  to skip a stage).

### 3. Run unbiased QM/MM MD

Leave `UMBRELLA_SAMPLING = False` and run:

```bash
python easyqmmm.py
```

This runs the staged MIN → NVT → NPT pipeline with plain QM/MM dynamics and
writes a trajectory, log, QM-region snapshots, and restart files.

### 4. Run biased QM/MM MD (umbrella sampling)

Set `UMBRELLA_SAMPLING = True` and configure:

- `UMBRELLA_MODE` — `"distance"` or `"distance_difference"`.
- `UMBRELLA_ATOMS` — 2 atoms (distance) or 4 atoms (distance-difference,
  e.g. a proton-transfer coordinate: `d(donor,H) - d(acceptor,H)`).
- `UMBRELLA_R1..R4`, `UMBRELLA_RK2`, `UMBRELLA_RK3` — same meaning as AMBER's
  `&rst` block (`r1/r2/r3/r4` in Å, `rk2/rk3` in kcal/mol/Å²).

One run = one umbrella window. To scan a series of windows the way you would
with a bash loop over sander `&rst` jobs, either edit `UMBRELLA_R2`/`UMBRELLA_R3`
between runs, or override them per submission with an environment variable:

```bash
for i in $(seq 2.0 0.1 3.0); do
    UMBRELLA_TARGET=$i python easyqmmm.py
done
```

Each window writes its own CV log (`umbrella_cv_win_<target>.dat`, columns:
step, time (ps), coordinate value in Å) ready for WHAM/MBAR analysis, plus its
own restart files.

### 5. Resuming a run

Point `INPCRD_FILE` at any written `restart_<label>_<stage>.rst7` file to resume
from that stage/checkpoint, exactly like resuming a normal AMBER restart.

---

## Output files

| File | Contents |
|---|---|
| `trajectory.dcd` | MM(+QM-injected) trajectory, NVT/NPT stages only |
| `simulation.log` | Step, time, energies, temperature, (volume), speed |
| `qm_region.xyz` | QM region + link-atom geometry snapshots over the run |
| `restart_<label>_<stage>.rst7` | AMBER-format restart (positions, velocities, box) per stage |
| `umbrella_cv_win_<target>.dat` | Reaction-coordinate time series (biased runs only) |
| ORCA scratch dir | `inpfile.xyz`, `ptchrg.xyz`, `orca.inp`, `orca.out`, `orca.engrad` (overwritten each QM step, ORCA backend only) |
| `psi4_output.dat` | Psi4 log (Psi4 backend only) |

---

## Notes & caveats

- **Link-atom scale factors are not universal.** The default (0.723) is tuned for
  a C(sp3)–C(sp3) cut capped with H. Retune `LINK_ATOM_SCALE_OVERRIDES` per bond
  type using `r_eq(QM–H) / r_eq(QM–MM)` for your specific cut.
- **ORCA must be given a full path**, not a bare `"orca"` resolved via `$PATH` —
  ORCA looks up its own MPI/shared-library environment relative to its own
  install path, and `%pal` parallel runs can otherwise fail silently.
- One run of the script corresponds to one umbrella window; window scanning is
  handled by re-invoking the script (see above), not internally.
- GPU acceleration applies to the MM/integration side only; QM step cost is
  dominated by Psi4/ORCA regardless of the OpenMM platform.

---

## Citation

If you use this workflow, please cite:

- Abdelgawwad, A. M. A.; Monari, A.; Tuñón, I.; Francés-Monerris, A. Spatial
  and Temporal Resolution of the Oxygen-Independent Photoinduced DNA
  Interstrand Cross-Linking by a Nitroimidazole Derivative. *J. Chem. Inf.
  Model.* **2022**, 62, 3239–3252 (source of the example system and the
  barrierless carbocation/nucleobase reactivity it reproduces).
- Walker, R. C.; Crowley, M. F.; Case, D. A. *J. Comput. Chem.* **2008**, 29,
  1019–1031 (AMBER additive QM/MM scheme).
- OpenMM: Eastman, P. et al. *PLOS Comput. Biol.* **2017**, 13, e1005659.
- Psi4 and/or ORCA, as appropriate to the backend used.
