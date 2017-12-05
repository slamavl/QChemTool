! fortranmodule.f95 
module fortranmodule
use omp_lib

contains
    subroutine test(x, y)
        real(8), dimension(:), intent(in)  :: x
        real(8), dimension(:), intent(out) :: y
        ! Code begins
        integer :: i, n
        integer :: num_threads
        n = size(x, 1)

        !$omp parallel do private(i) firstprivate(n) shared(x, y)
        do i = 1, n
            if (i == 1) then
                ! The if clause can be removed for serious use.
                ! It is here for debugging only.
                num_threads = OMP_get_num_threads()
                print *, 'num_threads running:', num_threads
            end if
            y(i) = sin(x(i)) + cos(x(i) + exp(x(i))) + log(x(i))
        end do
        !$omp end parallel do
    end subroutine test
	
	subroutine ftdc_omp(grid1,step1,origin1,rho1,grid2,step2,origin2,rho2,x) ! Posunuti molekul zahrnuto v posunuti origin
		IMPLICIT NONE
		integer,dimension(3), intent(in)        :: grid1,grid2         ! velikost mrize pro vypis hustoty
		real(8),dimension(3), intent(in)        :: origin1,origin2       ! pocatek mrize
		real(8),dimension(3,3), intent(in)      :: step1,step2         ! step(1,:) krok ve smeru prvni dimenze gridu (ve smeru X) atd.
		real(8),dimension(:,:,:), intent(in)	:: rho1,rho2          ! nabojova hustota
		real(8),intent(out)                     :: x                    ! interaction energy
		real(8),allocatable,dimension(:,:,:)	:: RR2X,RR2Y,RR2Z,Norm_mat		

		integer :: i,j,k, m,n,o
		real(8) :: norm,dV1,dV2
		real(8), dimension(3) :: drr,rr1,rr2,VecX,VecY,VecZ
		real(8), parameter :: MinDistance=0.001_8               ! Minimalni vzdalenost jednotlivych bodu pro vypocet interakcni energie
		
		Print *,'Starting fortran calculation: '
		allocate(RR2X(grid2(1),grid2(2),grid2(3)),RR2Y(grid2(1),grid2(2),grid2(3)),RR2Z(grid2(1),grid2(2),grid2(3)))

		do m=1,grid2(1)
		    do n=1,grid2(2)
		        do o=1,grid2(3)
		            rr2=origin2+(m-1)*step2(1,:)+(n-1)*step2(2,:)+(o-1)*step2(3,:)
					RR2X(m,n,o)=rr2(1)
					RR2Y(m,n,o)=rr2(2)
					RR2Z(m,n,o)=rr2(3)
		        enddo
		    enddo
		enddo

		Print *,'Starting parallel calculation: '
		Print *,'Pocet jader je: ',OMP_GET_NUM_THREADS()

		!$OMP PARALLEL DEFAULT(NONE) PRIVATE(i,j,k,rr1,m,n,o,Norm_mat) FIRSTPRIVATE(RR2X,RR2Y,RR2Z) SHARED(x,grid1,grid2,step1,step2,origin1,origin2,rho1,rho2)
		x=0.0_8
		allocate(Norm_mat(grid2(1),grid2(2),grid2(3)))
		Print *,'Pocet jader je: ',OMP_GET_NUM_THREADS()
		!$OMP DO SCHEDULE(DYNAMIC) REDUCTION (+:x)
		do i=1,grid1(1)
		    do j=1,grid1(2)
		        do k=1,grid1(3)
		            rr1=origin1+(i-1)*step1(1,:)+(j-1)*step1(2,:)+(k-1)*step1(3,:)
		            Norm_mat=sqrt( (RR2X-rr1(1))**2 + (RR2Y-rr1(2))**2 + (RR2Z-rr1(3))**2 )
					x = x + rho1(i,j,k)*sum(rho2/Norm_mat)
		        enddo
		    enddo
		    print '(I0,a,I0,a,I0)',i,'/',grid1(1),' From thread: ',OMP_GET_THREAD_NUM()
		    
		enddo
		!$OMP END DO
		!$OMP END PARALLEL
		
		vecX=step1(1,:)
		vecY=step1(2,:)
		VecZ=(/VecX(2)*VecY(3)-VecX(3)*VecY(2),VecX(3)*VecY(1)-VecX(1)*VecY(3),VecX(1)*VecY(2)-VecX(2)*VecY(1)/)
		dV1=dot_product(VecZ,step1(3,:))
		vecX=step2(1,:)
		vecY=step2(2,:)
		VecZ=(/VecX(2)*VecY(3)-VecX(3)*VecY(2),VecX(3)*VecY(1)-VecX(1)*VecY(3),VecX(1)*VecY(2)-VecX(2)*VecY(1)/)
		dV2=dot_product(VecZ,step2(3,:))
		x=x*dV1*dV2
	end subroutine ftdc_omp

	subroutine fTDC_old_omp(grid1,step1,origin1,rho1,grid2,step2,origin2,rho2,x) ! Posunuti molekul zahrnuto v posunuti origin
		IMPLICIT NONE
		integer,dimension(3), intent(in)        :: grid1,grid2         ! velikost mrize pro vypis hustoty
		real(8),dimension(3), intent(in)        :: origin1,origin2       ! pocatek mrize
		real(8),dimension(3,3), intent(in)      :: step1,step2         ! step(1,:) krok ve smeru prvni dimenze gridu (ve smeru X) atd.
		real(8),dimension(:,:,:), intent(in)	:: rho1,rho2          ! nabojova hustota
		real(8),intent(out)                     :: x                    ! interaction energy
		
		integer :: i,j,k, m,n,o
		real(8) :: norm,dV1,dV2
		real(8), dimension(3) :: drr,rr1,rr2,VecX,VecY,VecZ
		real(8), parameter :: MinDistance=0.001_8               ! Minimalni vzdalenost jednotlivych bodu pro vypocet interakcni energie
		
		Print *,'Starting parallel calculation: '
		Print *,'Pocet jader je: ',OMP_GET_NUM_THREADS()
		!$OMP PARALLEL DEFAULT(NONE) PRIVATE(i,j,k,rr1,m,n,o,rr2,drr,norm) SHARED (x,grid1,grid2,step1,step2,origin1,origin2,rho1,rho2)
		x=0.0_8
		Print *,'Pocet jader je: ',OMP_GET_NUM_THREADS()
		!$OMP DO SCHEDULE(DYNAMIC) REDUCTION (+:x)
		do i=1,grid1(1)
		    do j=1,grid1(2)
		        do k=1,grid1(3)
		            rr1=origin1+(i-1)*step1(1,:)+(j-1)*step1(2,:)+(k-1)*step1(3,:)
		            do m=1,grid2(1)
		                do n=1,grid2(2)
		                    do o=1,grid2(3)
		                        rr2=origin2+(m-1)*step2(1,:)+(n-1)*step2(2,:)+(o-1)*step2(3,:)
		                        drr=rr2-rr1
		                        norm=sqrt(dot_product(drr,drr))
		                        if (norm<=MinDistance) CYCLE
		                        x = x + rho1(i,j,k)*rho2(m,n,o)/norm
		                    enddo
		                enddo
		            enddo
		        enddo
		    enddo
		    print '(I0,a,I0,a,I0)',i,'/',grid1(1),' From thread: ',OMP_GET_THREAD_NUM()
		    
		enddo
		!$OMP END DO
		!$OMP END PARALLEL
		
		vecX=step1(1,:)
		vecY=step1(2,:)
		VecZ=(/VecX(2)*VecY(3)-VecX(3)*VecY(2),VecX(3)*VecY(1)-VecX(1)*VecY(3),VecX(1)*VecY(2)-VecX(2)*VecY(1)/)
		dV1=dot_product(VecZ,step1(3,:))
		vecX=step2(1,:)
		vecY=step2(2,:)
		VecZ=(/VecX(2)*VecY(3)-VecX(3)*VecY(2),VecX(3)*VecY(1)-VecX(1)*VecY(3),VecX(1)*VecY(2)-VecX(2)*VecY(1)/)
		dV2=dot_product(VecZ,step2(3,:))
		x=x*dV1*dV2
	end subroutine fTDC_old_OMP

	
end module fortranmodule


