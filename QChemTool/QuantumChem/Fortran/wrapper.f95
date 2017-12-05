module w_fortranmodule
use fortranmodule, only: test,fTDC_OMP,fTDC_old_OMP
implicit none

contains
    subroutine w_test(x, y, ysize)
        real(8), intent(in), dimension(ysize)   :: x
        real(8), intent(out), dimension(ysize)  :: y
        integer                                 :: ysize
        call test(x, y)
    end subroutine w_test

	subroutine tdc(grid1,step1,origin1,rho1,grid2,step2,origin2,rho2,x,s1x,s1y,s1z,s2x,s2y,s2z)
		integer,dimension(3), intent(in)                            :: grid1,grid2         ! velikost mrize pro vypis hustoty
		real(8),dimension(3), intent(in)                            :: origin1,origin2       ! pocatek mrize
		real(8),dimension(3,3), intent(in)                          :: step1,step2         ! step(1,:) krok ve smeru prvni dimenze gridu (ve smeru X) atd.
		real(8),dimension(s1x,s1y,s1z), intent(in)	:: rho1                ! nabojova hustota
		real(8),dimension(s2x,s2y,s2z), intent(in)	:: rho2          ! nabojova hustota
		real(8),intent(out)                                         :: x                    ! interaction energy
		integer                                                     :: s1x,s1y,s1z,s2x,s2y,s2z
		call ftdc_omp(grid1,step1,origin1,rho1,grid2,step2,origin2,rho2,x)
	end subroutine tdc
	
	subroutine tdc_old(grid1,step1,origin1,rho1,grid2,step2,origin2,rho2,x,s1x,s1y,s1z,s2x,s2y,s2z)
		integer,dimension(3), intent(in)                            :: grid1,grid2         ! velikost mrize pro vypis hustoty
		real(8),dimension(3), intent(in)                            :: origin1,origin2       ! pocatek mrize
		real(8),dimension(3,3), intent(in)                          :: step1,step2         ! step(1,:) krok ve smeru prvni dimenze gridu (ve smeru X) atd.
		real(8),dimension(s1x,s1y,s1z), intent(in)	:: rho1                ! nabojova hustota
		real(8),dimension(s2x,s2y,s2z), intent(in)	:: rho2          ! nabojova hustota
		real(8),intent(out)                                         :: x                    ! interaction energy
		integer                                                     :: s1x,s1y,s1z,s2x,s2y,s2z
		call fTDC_old_omp(grid1,step1,origin1,rho1,grid2,step2,origin2,rho2,x)
	end subroutine tdc_old
end module w_fortranmodule
