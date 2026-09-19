!> Adapts the Pyrometheus-generated heterogeneous kinetics to the quantities the ghost-cell surface solve works in.
!!
!! Pyrometheus works in activity concentrations over every species the interface
!! couples -- its own surface species first, then the adjacent phases' -- while
!! the surface solve works in gas-phase mass fractions and a density. The two
!! conversions are all that lives here; none of the chemistry does.
!!
!! This module is static. Everything mechanism-dependent (how many species each
!! phase has, and where each phase's block starts) is a parameter of the
!! generated module.
#:include 'macros.fpp'

!> @brief Adapter between the ghost-cell surface solve and the Pyrometheus-generated heterogeneous kinetics.
module m_surface_thermochem

    use m_precision_select, only: wp
    use m_thermochem, only: molecular_weights
    use m_surface_thermochem_pyro, only: num_surface_species, num_coupled_species, num_coupled_gas_species, coupled_gas_offset, &
        & get_site_concentrations, get_surface_net_heat_release_rate, &
        & get_surface_net_production_rates_coupled => get_surface_net_production_rates

    implicit none

    private
    public :: get_surface_net_production_rates
    public :: get_surface_reaction_heat_flux

contains

    !> Activity concentration of every coupled species, from the gas-phase state at the wall. A bulk species is a pure condensed
    !! phase, so its activity is one rather than its molar density; a surface species sits at its site concentration, which is why
    !! the coverages come in separately.
    subroutine s_coupled_concentrations(density, mass_fractions, coverages, concentrations)

        $:GPU_ROUTINE(parallelism='[seq]')

        real(wp), intent(in)  :: density
        real(wp), intent(in)  :: mass_fractions(num_coupled_gas_species)
        real(wp), intent(in)  :: coverages(num_surface_species)
        real(wp), intent(out) :: concentrations(num_coupled_species)
        integer               :: k

        concentrations = 1._wp

        call get_site_concentrations(coverages, concentrations(1:num_surface_species))

        do k = 1, num_coupled_gas_species
            concentrations(coupled_gas_offset + k) = density*mass_fractions(k)/molecular_weights(k)
        end do

    end subroutine s_coupled_concentrations

    !> Production rate of each gas species at the reacting surface, in kmol/m^2-s.
    subroutine get_surface_net_production_rates(density, temperature, mass_fractions, omega_s)

        $:GPU_ROUTINE(parallelism='[seq]')

        real(wp), intent(in)  :: density
        real(wp), intent(in)  :: temperature
        real(wp), intent(in)  :: mass_fractions(num_coupled_gas_species)
        real(wp), intent(out) :: omega_s(num_coupled_gas_species)
        real(wp)              :: concentrations(num_coupled_species)
        real(wp)              :: omega(num_coupled_species)
        real(wp)              :: coverages(num_surface_species)

        ! The surface solve carries no coverage state, so this value has to be
        ! arbitrary. It is only sound because the mechanism is checked, when the
        ! case is read, to have no surface species in any reaction and no
        ! coverage-dependent rate -- so no rate expression reads it.
        coverages = 1._wp/real(num_surface_species, wp)

        call s_coupled_concentrations(density, mass_fractions, coverages, concentrations)
        call get_surface_net_production_rates_coupled(temperature, concentrations, coverages, omega)

        omega_s = omega(coupled_gas_offset + 1:coupled_gas_offset + num_coupled_gas_species)

    end subroutine get_surface_net_production_rates

    !> Heat released by the surface reactions, in W/m^2, positive when exothermic.
    subroutine get_surface_reaction_heat_flux(density, temperature, mass_fractions, q_rxn)

        $:GPU_ROUTINE(parallelism='[seq]')

        real(wp), intent(in)  :: density
        real(wp), intent(in)  :: temperature
        real(wp), intent(in)  :: mass_fractions(num_coupled_gas_species)
        real(wp), intent(out) :: q_rxn
        real(wp)              :: concentrations(num_coupled_species)
        real(wp)              :: coverages(num_surface_species)

        ! Arbitrary for the same reason as above.
        coverages = 1._wp/real(num_surface_species, wp)

        call s_coupled_concentrations(density, mass_fractions, coverages, concentrations)
        call get_surface_net_heat_release_rate(temperature, concentrations, coverages, q_rxn)

    end subroutine get_surface_reaction_heat_flux

end module m_surface_thermochem
