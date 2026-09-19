"""Guards on the surface-mechanism half of the chemistry toolchain.

The kinetics itself is Pyrometheus-generated and tested there, including the rate
forms it refuses to translate. What is checked here is the narrower thing MFC owns:
which mechanisms the ghost-cell surface solve can represent at all. It carries no
coverage state, so a mechanism whose rates depend on one would be evaluated at a
coverage this code invented -- a wrong number rather than a message.

Cantera's own ptcombust.yaml is the fixture because it carries every kind at once:
19 interface-Arrhenius reactions, 5 sticking-coefficient ones, and 2 with coverage
dependence. A synthetic mechanism would only prove Cantera parses what we wrote.
"""

import os

import pytest

from mfc import common

ct = pytest.importorskip("cantera", reason="the surface codegen needs Cantera")

# What Pyrometheus translates. Kept as data so a test failure names the mismatch
# rather than a bare isinstance.
SUPPORTED = (ct.ArrheniusRate, ct.InterfaceArrheniusRate, ct.StickingArrheniusRate)


# Exactly what src/simulation/m_surface_thermochem.fpp names in its `use` statement.
# Both the generated module and the no-surface stub have to carry all of it, and a
# missing one is a link error at the end of a long build rather than here.
ADAPTER_IMPORTS = (
    "num_surface_species",
    "num_coupled_species",
    "num_coupled_gas_species",
    "coupled_gas_offset",
    "get_site_concentrations",
    "get_surface_net_production_rates",
    "get_surface_net_heat_release_rate",
)


def _coverage(rate):
    return dict(getattr(rate, "coverage_dependencies", {}) or {})


@pytest.fixture(scope="module")
def ptcombust():
    return ct.Interface("ptcombust.yaml", "Pt_surf")


def test_a_rate_form_pyrometheus_cannot_translate_is_refused_there():
    """The one guard MFC no longer carries: it belongs to the generator.

    Every Cantera surface rate class exposes pre_exponential_factor, so an unsupported
    one reads as a plausible Arrhenius rate rather than raising. This checks the
    dependency still says no, because nothing on the MFC side would."""
    import pymbolic.primitives as p
    from pyrometheus import chem_expr

    surface = ct.Interface("ptcombust.yaml", "Pt_surf")
    reaction = next(r for r in surface.reactions() if isinstance(r.rate, ct.StickRateBase))
    assert hasattr(reaction.rate, "pre_exponential_factor"), "the weak guard this test rules out"

    # A sticking-Arrhenius rate is translated; its Blowers-Masel sibling is not, and
    # both are StickRateBase, so only the exact class separates them.
    coverages = [p.Variable(f"theta_{k}") for k in range(surface.n_species)]
    chem_expr.surface_rate_coefficient_expr(surface, reaction, p.Variable("t"), coverages)


def test_coverage_dependent_rates_are_refused(ptcombust):
    """The surface solve has no coverage to evaluate them at."""
    from mfc.run.input import MFCInputFile

    covered = [r for r in ptcombust.reactions() if _coverage(r.rate)]
    assert covered, "ptcombust.yaml is expected to carry coverage-dependent reactions"

    case = MFCInputFile("case.py", ".", {})
    with pytest.raises(common.MFCException, match="coverage"):
        case.validate_surface_mechanism(ct.Solution("ptcombust.yaml", "gas"), ptcombust)


def test_the_shipped_carbon_mechanism_is_translatable_end_to_end():
    """The reacting-surface example must keep generating. This is the case the goldens rely on,
    so a guard that turns it away breaks the feature rather than protecting it."""
    path = os.path.join(common.MFC_ROOT_DIR, "examples", "2D_ibm_reacting_surface", "carbon_surface_bradley_11species.yaml")
    if not os.path.isfile(path):
        pytest.skip("reacting-surface example not present")

    surface = ct.Interface(path, "carbon_surface")
    assert surface.n_reactions > 0

    for reaction in surface.reactions():
        assert isinstance(reaction.rate, SUPPORTED), reaction.equation
        assert not _coverage(reaction.rate), reaction.equation
        assert not (set(reaction.reactants) | set(reaction.products)) & set(surface.species_names), reaction.equation


def test_surface_site_species_in_stoichiometry_are_refused_with_a_reason():
    """ptcombust is a coverage-based mechanism: its reactions consume and produce adsorbed
    sites (PT(S), O(S)). Pyrometheus handles those, but the surface solve does not evolve
    the coverages they change, so the case has to be turned away rather than run at a
    coverage nobody chose."""
    from mfc.run.input import MFCInputFile

    case = MFCInputFile("case.py", ".", {})
    surface = ct.Interface("ptcombust.yaml", "Pt_surf")

    with pytest.raises(common.MFCException, match="surface-site species|coverage"):
        case.validate_surface_mechanism(ct.Solution("ptcombust.yaml", "gas"), surface)


def test_the_shipped_carbon_mechanism_generates_compilable_fortran():
    """The end-to-end counterpart: the example's mechanism must produce a module carrying
    everything src/simulation/m_surface_thermochem.fpp imports from it."""
    from mfc.run.input import MFCInputFile

    path = os.path.join(common.MFC_ROOT_DIR, "examples", "2D_ibm_reacting_surface", "carbon_surface_bradley_11species.yaml")
    if not os.path.isfile(path):
        pytest.skip("reacting-surface example not present")

    # Through get_cantera_surface, not a hand-built ct.Interface: the mechanism declares adjacent
    # phases that have to be resolved and loaded, which is exactly what that method is for, so this
    # covers the resolution path as well as the generator.
    case = MFCInputFile(
        "case.py",
        os.path.dirname(path),
        {
            "chemistry": "T",
            "cantera_file": os.path.join(os.path.dirname(path), "carbon_gasphase_reduced_gri11.yaml"),
            "surface_cantera_file": path,
            "surface_phase": "carbon_surface",
        },
    )
    import pyrometheus as pyro

    gas = case.get_cantera_solution()
    surface = case.get_cantera_surface()
    assert surface is not None, "the example's surface mechanism failed to resolve"
    case.validate_surface_mechanism(gas, surface)

    code = pyro.FortranCodeGenerator().generate_surface("m_surface_thermochem_pyro", surface, gas_module_name="m_thermochem")

    assert "module m_surface_thermochem_pyro" in code
    for name in ADAPTER_IMPORTS:
        assert name in code, f"the adapter imports {name}, which the generator did not emit"


def test_a_case_without_a_surface_mechanism_still_generates_a_module():
    """Chemistry without surface reactions must still compile: the adapter is built
    unconditionally, so every name it imports has to exist in the no-surface form too.

    Pyrometheus cannot supply them -- it has no interface to generate from -- which is
    the whole reason MFC writes this one out by hand."""
    from mfc.run.input import MFCInputFile

    code = MFCInputFile("case.py", ".", {}).generate_surface_stub(ct.Solution("h2o2.yaml"), "real(dp)")

    assert "module m_surface_thermochem_pyro" in code
    for name in ADAPTER_IMPORTS:
        assert name in code, f"the adapter imports {name}, which the stub does not define"
