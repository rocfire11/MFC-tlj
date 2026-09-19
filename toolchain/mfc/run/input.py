import dataclasses
import glob
import json
import os
import typing

from .. import case_validator, common
from ..case import Case

# Note: pyrometheus and cantera are imported lazily in the methods that need them
# to avoid slow startup times for commands that don't use chemistry features
# Note: build is imported lazily to avoid circular import with build.py
from ..printer import cons
from ..state import ARG, ARGS, gpuConfigOptions


@dataclasses.dataclass(init=False)
class MFCInputFile(Case):
    filename: str
    dirpath: str

    def __init__(self, filename: str, dirpath: str, params: dict) -> None:
        super().__init__(params)
        self.filename = filename
        self.dirpath = dirpath

    def generate_inp(self, target) -> None:
        from .. import build

        target = build.get_target(target)

        # Save .inp input file
        common.file_write(f"{self.dirpath}/{target.name}.inp", self.get_inp(target))

    def __save_fpp(self, target, contents: str) -> None:
        inc_dir = os.path.join(target.get_staging_dirpath(self), "include", target.name)
        common.create_directory(inc_dir)

        fpp_path = os.path.join(inc_dir, "case.fpp")

        cons.print("Writing a (new) custom case.fpp file.")
        common.file_write(fpp_path, contents, True)

    def get_cantera_solution(self):
        # Lazy import to avoid slow startup for commands that don't need chemistry
        import cantera as ct

        if self.params.get("chemistry", "F") == "T":
            cantera_file = self.params["cantera_file"]
            candidates = [
                cantera_file,
                os.path.join(self.dirpath, cantera_file),
                os.path.join(common.MFC_MECHANISMS_DIR, cantera_file),
            ]
        else:
            # Chemistry is off — return a dummy solution so MFC still compiles.
            cantera_file = "h2o2.yaml"
            candidates = [cantera_file]

        for candidate in candidates:
            try:
                return ct.Solution(candidate)
            except Exception as e:
                cons.print(f"[dim]  Cantera: skipping '{candidate}': {e}[/dim]")
                continue

        raise common.MFCException(f"Cantera file '{cantera_file}' not found. Searched: {', '.join(candidates)}.")

    def get_cantera_surface(self):
        # Lazy import to avoid slow startup for commands that don't need chemistry
        import cantera as ct
        import yaml

        surface_file = self.params.get("surface_cantera_file")
        surface_phase = self.params.get("surface_phase")

        if surface_file is None and surface_phase is None:
            return None

        if surface_file is None or surface_phase is None:
            raise common.MFCException("surface_cantera_file and surface_phase must be specified together.")

        candidates = [
            surface_file,
            os.path.join(self.dirpath, surface_file),
            os.path.join(common.MFC_MECHANISMS_DIR, surface_file),
        ]

        gas = self.get_cantera_solution()

        # Why every failure is recorded and the loop continues rather than raising on the spot: a file
        # of the same name sitting in the case directory without the requested phase must not stop the
        # copy in MFC_MECHANISMS_DIR from being tried.
        reasons = []

        for candidate in candidates:
            if not os.path.isfile(candidate):
                reasons.append(f"{candidate}: no such file")
                continue

            try:
                with open(candidate, "r", encoding="utf-8") as stream:
                    mechanism = yaml.safe_load(stream)

                phases = mechanism.get("phases", [])

                interface_data = None
                for phase in phases:
                    if phase.get("name") == surface_phase:
                        interface_data = phase
                        break

                if interface_data is None:
                    found = ", ".join(str(phase.get("name")) for phase in phases) or "none"
                    reasons.append(f"{candidate}: phase '{surface_phase}' not found (has: {found})")
                    continue

                adjacent_names = interface_data.get("adjacent-phases", [])

                adjacent = []

                for phase_name in adjacent_names:
                    if phase_name == gas.name:
                        adjacent.append(gas)
                    else:
                        adjacent.append(ct.Solution(candidate, phase_name))

                return ct.Interface(
                    candidate,
                    surface_phase,
                    adjacent=adjacent,
                )

            except Exception as e:
                cons.print(f"[dim]  Cantera: skipping surface mechanism " f"'{candidate}': {e}[/dim]")
                reasons.append(f"{candidate}: {e}")
                continue

        raise common.MFCException(f"Cantera surface file '{surface_file}' with phase " f"'{surface_phase}' could not be loaded. Tried:\n  " + "\n  ".join(reasons))

    def validate_surface_mechanism(self, sol, surface) -> None:
        """Reject a surface mechanism the surface solve cannot represent.

        Pyrometheus refuses the rate forms it cannot translate. What is checked here
        is narrower and belongs to MFC: the ghost-cell solve carries no surface
        coverage state, so a mechanism whose rates depend on one would be evaluated
        at a coverage this code invented. Better a message than a plausible number.
        """
        # Lazy import to avoid slow startup for commands that don't need chemistry
        from pyrometheus.chem_expr import surface_phase_kind

        gas_species = list(sol.species_names)

        # The generated code addresses the gas species by position in the coupled
        # ordering, so the two mechanisms have to agree on the list and its order,
        # not merely on the names present. Which adjacent phase is the gas one is
        # asked of Cantera rather than assumed from its name.
        for phase in surface.adjacent.values():
            if surface_phase_kind(phase) != "gas":
                continue

            if list(phase.species_names) != gas_species:
                raise common.MFCException(
                    "The surface mechanism's gas phase and the case's gas mechanism must "
                    "list the same species in the same order; the generated code pairs "
                    f"them by position. Case: {gas_species}. "
                    f"Surface: {list(phase.species_names)}."
                )

        for number, reaction in enumerate(surface.reactions(), start=1):
            for name in set(reaction.reactants) | set(reaction.products):
                if name in surface.species_names:
                    raise common.MFCException(
                        f"Surface reaction {number} ('{reaction.equation}') consumes or "
                        f"produces the surface-site species '{name}'. The surface solve "
                        "carries no coverage state to evolve, so such a mechanism is not "
                        "supported."
                    )

            coverage = dict(getattr(reaction.rate, "coverage_dependencies", {}) or {})
            if coverage:
                raise common.MFCException(
                    f"Surface reaction {number} ('{reaction.equation}') declares coverage "
                    f"dependencies for {', '.join(sorted(coverage))}. The surface solve "
                    "carries no coverage state, so the rate would be evaluated at an "
                    "invented one."
                )

    def generate_surface_stub(self, sol, real_type, directive_str=None) -> str:
        """Stand in for the generated surface kinetics when a case has no surface mechanism.

        The adapter in m_surface_thermochem is compiled either way, so the names it
        uses have to exist. Pyrometheus cannot supply them without an interface to
        generate from, which is the whole of why this is written out by hand.
        """
        if directive_str == "mp":
            gpu_routine_define = "#define GPU_ROUTINE(name) !$omp declare target"
        elif directive_str == "acc":
            gpu_routine_define = "#define GPU_ROUTINE(name) !$acc routine seq"
        else:
            gpu_routine_define = "#define GPU_ROUTINE(name) ! name"

        num_species = len(sol.species_names)

        return f"""! This file is automatically generated by the MFC toolchain.
! Do not edit manually.

{gpu_routine_define}

module m_surface_thermochem_pyro

    use m_thermochem, only: sp, dp

    implicit none

    ! One nominal site so that the adapter's arrays are not zero-sized.
    integer, parameter :: num_surface_species = 1
    integer, parameter :: num_coupled_species = {num_species + 1}
    integer, parameter :: num_coupled_gas_species = {num_species}
    integer, parameter :: coupled_surface_offset = 0
    integer, parameter :: coupled_gas_offset = 1

contains

    subroutine get_site_concentrations(coverages, concentrations)

        GPU_ROUTINE(get_site_concentrations)

        {real_type}, intent(in) :: coverages(num_surface_species)
        {real_type}, intent(out) :: concentrations(num_surface_species)

        concentrations = 0*coverages

    end subroutine get_site_concentrations

    subroutine get_surface_net_production_rates(temperature, concentrations, &
            coverages, omega)

        GPU_ROUTINE(get_surface_net_production_rates)

        {real_type}, intent(in) :: temperature
        {real_type}, intent(in) :: concentrations(num_coupled_species)
        {real_type}, intent(in) :: coverages(num_surface_species)
        {real_type}, intent(out) :: omega(num_coupled_species)

        omega = 0*temperature*concentrations*coverages(1)

    end subroutine get_surface_net_production_rates

    subroutine get_surface_net_heat_release_rate(temperature, concentrations, &
            coverages, q)

        GPU_ROUTINE(get_surface_net_heat_release_rate)

        {real_type}, intent(in) :: temperature
        {real_type}, intent(in) :: concentrations(num_coupled_species)
        {real_type}, intent(in) :: coverages(num_surface_species)
        {real_type}, intent(out) :: q

        q = 0*temperature*concentrations(1)*coverages(1)

    end subroutine get_surface_net_heat_release_rate

end module m_surface_thermochem_pyro
"""

    def generate_fpp(self, target) -> None:
        # Lazy import to avoid slow startup for commands that don't need chemistry
        import pyrometheus as pyro

        if target.isDependency:
            return

        cons.print("Generating [magenta]case.fpp[/magenta].")
        cons.indent()

        # Case FPP file
        self.__save_fpp(target, self.get_fpp(target))

        # (Thermo)Chemistry source file
        modules_dir = os.path.join(target.get_staging_dirpath(self), "modules", target.name)
        common.create_directory(modules_dir)

        # Determine the real type based on the single precision flag
        real_type = "real(sp)" if (ARG("single") or ARG("mixed")) else "real(dp)"

        if ARG("gpu") == gpuConfigOptions.MP.value:
            directive_str = "mp"
        elif ARG("gpu") == gpuConfigOptions.ACC.value:
            directive_str = "acc"
        else:
            directive_str = None

        # Write the generated Fortran code to the m_thermochem.f90 file with the chosen precision
        sol = self.get_cantera_solution()
        surface = self.get_cantera_surface() if target.name == "simulation" else None
        if surface is not None:
            cons.print(f"Loaded Cantera surface phase '{surface.name}' " f"with {surface.n_reactions} reaction(s).")

        thermochem_code = pyro.FortranCodeGenerator().generate("m_thermochem", sol, pyro.CodeGenerationOptions(scalar_type=real_type, directive_offload=directive_str))

        common.file_write(os.path.join(modules_dir, "m_thermochem.f90"), thermochem_code, True)

        if target.name == "simulation":
            # The heterogeneous kinetics is Pyrometheus-generated like the gas-phase
            # thermochemistry; src/simulation/m_surface_thermochem.fpp adapts it to
            # the mass fractions and density the surface solve works in.
            if surface is None:
                surface_thermochem_code = self.generate_surface_stub(sol, real_type, directive_str)
            else:
                self.validate_surface_mechanism(sol, surface)
                surface_thermochem_code = pyro.FortranCodeGenerator().generate_surface(
                    "m_surface_thermochem_pyro",
                    surface,
                    gas_module_name="m_thermochem",
                    opts=pyro.CodeGenerationOptions(scalar_type=real_type, directive_offload=directive_str),
                )

            common.file_write(
                os.path.join(modules_dir, "m_surface_thermochem_pyro.f90"),
                surface_thermochem_code,
                True,
            )

            if surface is not None:
                cons.print(f"Generated m_surface_thermochem_pyro.f90 with " f"{surface.n_reactions} surface reaction(s).")

        cons.unindent()

    def validate_constraints(self, target) -> None:
        """Validate case parameter constraints for a given target stage"""
        from .. import build

        target_obj = build.get_target(target)
        stage = target_obj.name

        try:
            warnings = case_validator.validate_case_constraints(self.params, stage)
        except case_validator.CaseConstraintError as e:
            raise common.MFCException(f"Case validation failed for {stage}:\n{e}") from e

        if warnings:
            cons.print()
            cons.print("[bold yellow]Physics warnings:[/bold yellow]")
            for warning in warnings:
                cons.print(f"  [yellow]- {warning}[/yellow]")
            cons.print()

    # Generate case.fpp & [target.name].inp
    def generate(self, target) -> None:
        # Validate constraints before generating input files
        self.validate_constraints(target)
        self.generate_inp(target)
        cons.print()
        self.generate_fpp(target)

    def clean(self, _targets) -> None:
        from .. import build

        targets = [build.get_target(target) for target in _targets]

        files = set()
        dirs = set()

        files = set(["equations.dat", "run_time.inf", "time_data.dat", "io_time_data.dat", "fort.1", "pre_time_data.dat"] + [f"{target.name}.inp" for target in targets])

        if build.PRE_PROCESS in targets:
            files = files | set(glob.glob(os.path.join(self.dirpath, "D", "*.000000.dat")))
            dirs = dirs | set(glob.glob(os.path.join(self.dirpath, "p_all", "p*", "0")))

        if build.SIMULATION in targets:
            restarts = set(glob.glob(os.path.join(self.dirpath, "restart_data", "*.dat")))
            restarts = restarts - set(glob.glob(os.path.join(self.dirpath, "restart_data", "lustre_0.dat")))
            restarts = restarts - set(glob.glob(os.path.join(self.dirpath, "restart_data", "lustre_*_cb.dat")))

            Ds = set(glob.glob(os.path.join(self.dirpath, "D", "*.dat")))
            Ds = Ds - set(glob.glob(os.path.join(self.dirpath, "D", "*.000000.dat")))

            files = files | restarts
            files = files | Ds

        if build.POST_PROCESS in targets:
            dirs.add("silo_hdf5")

        for relfile in files:
            filepath = relfile if os.path.isfile(relfile) else os.path.join(self.dirpath, relfile)
            common.delete_file(filepath)

        for reldir in dirs:
            dirpath = reldir if os.path.isdir(reldir) else os.path.join(self.dirpath, reldir)
            common.delete_directory(dirpath)


# Load the input file
def load(filepath: str = None, args: typing.List[str] = None, empty_data: dict = None, do_print: bool = True) -> MFCInputFile:
    if not filepath:
        if empty_data is None:
            raise common.MFCException("Please provide an input file.")

        input_file = MFCInputFile("empty.py", "empty.py", empty_data)
        input_file.validate_params()
        return input_file

    filename: str = filepath.strip()

    if do_print:
        cons.print(f"Acquiring [bold magenta]{filename}[/bold magenta]...")

    dirpath: str = os.path.abspath(os.path.dirname(filename))
    dictionary: dict = {}

    if not os.path.exists(filename):
        raise common.MFCException(f"Input file '{filename}' does not exist. Please check the path is valid.")

    if filename.endswith(".py"):
        (json_str, err) = common.get_py_program_output(filename, ["--mfc", json.dumps(ARGS())] + (args or []))

        if err != 0:
            raise common.MFCException(f"Input file {filename} terminated with a non-zero exit code. Please make sure running the file doesn't produce any errors.")
    elif filename.endswith(".json"):
        json_str = common.file_read(filename)
    elif filename.endswith((".yaml", ".yml")):
        import yaml

        with open(filename, "r") as f:
            dictionary = yaml.safe_load(f)
        json_str = json.dumps(dictionary)
    else:
        raise common.MFCException("Unrecognized input file format. Supported: .py, .json, .yaml, .yml. Please check the README and sample cases in the examples directory.")

    try:
        dictionary = json.loads(json_str)
    except Exception as exc:
        raise common.MFCException(f"Input file {filename} did not produce valid JSON. It should only print the case dictionary.\n\n{exc}\n")

    input_file = MFCInputFile(filename, dirpath, dictionary)
    input_file.validate_params(f"Input file {filename}")
    return input_file


load.CACHED_MFCInputFile = None
