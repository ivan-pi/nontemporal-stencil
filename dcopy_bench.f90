!> dcopy_bench.f90
!! ---------------------------------------------------------------------------
!! Standalone benchmark comparing standard (temporal) stores against
!! non-temporal / streaming stores for a BLAS Level-1 DCOPY, y(1:n) = x(1:n).
!!
!! Four variants are timed:
!!    * array syntax        -- plain Fortran  y = x
!!    * memcpy              -- C library memcpy   (dcopy_memcpy_)
!!    * standard store      -- vectorised temporal stores (dcopy_store_)
!!    * non-temporal store  -- vectorised streaming stores (dcopy_nt_)
!!
!! The C kernels are declared with explicit interfaces but WITHOUT bind(C):
!! they are reached through the classic Fortran name-mangling ABI (trailing
!! underscore, by-reference arguments), i.e. linked as dcopy_store_ etc.,
!! just like a legacy BLAS library.
!!
!! Usage:  ./dcopy_bench [n] [iterations]
!!   n            elements per vector (default 8000000 -> 64 MB per vector)
!!   iterations   timed copies per variant (default 50)
!!
!! Pick n so the two vectors are comfortably larger than the last-level
!! cache; that is where non-temporal stores pull ahead.
!! ---------------------------------------------------------------------------
module dcopy_bench_mod
  use, intrinsic :: iso_fortran_env, only: real64, int64, output_unit
  implicit none
  private
  public :: real64, copy_kernel, dcopy_array, dcopy_store, dcopy_memcpy, &
            dcopy_nt, run_bench

  !> Common signature shared by every copy kernel.
  abstract interface
     subroutine copy_kernel(n, x, y)
       import :: real64
       integer,      intent(in)  :: n
       real(real64), intent(in)  :: x(n)
       real(real64), intent(out) :: y(n)
     end subroutine copy_kernel
  end interface

  !> Explicit interfaces for the C kernels.  No bind(C) -> the linker sees the
  !! name-mangled symbols dcopy_store_, dcopy_memcpy_, dcopy_nt_.
  interface
     subroutine dcopy_store(n, x, y)
       import :: real64
       integer,      intent(in)  :: n
       real(real64), intent(in)  :: x(n)
       real(real64), intent(out) :: y(n)
     end subroutine dcopy_store

     subroutine dcopy_memcpy(n, x, y)
       import :: real64
       integer,      intent(in)  :: n
       real(real64), intent(in)  :: x(n)
       real(real64), intent(out) :: y(n)
     end subroutine dcopy_memcpy

     subroutine dcopy_nt(n, x, y)
       import :: real64
       integer,      intent(in)  :: n
       real(real64), intent(in)  :: x(n)
       real(real64), intent(out) :: y(n)
     end subroutine dcopy_nt
  end interface

contains

  !> The array-syntax variant, wrapped so it matches copy_kernel.
  subroutine dcopy_array(n, x, y)
    integer,      intent(in)  :: n
    real(real64), intent(in)  :: x(n)
    real(real64), intent(out) :: y(n)
    y = x
  end subroutine dcopy_array

  !> Copy y = x by handing contiguous chunks to `kernel`.  Built with
  !! -fopenmp the chunks run across threads -- and it is precisely the
  !! resulting memory-bandwidth pressure that lets non-temporal stores show
  !! their advantage; without -fopenmp the loop is simply serial.  Every
  !! variant goes through here, so the comparison stays apples-to-apples.
  subroutine par_copy(kernel, n, x, y)
    procedure(copy_kernel)              :: kernel
    integer,              intent(in)    :: n
    real(real64),         intent(in)    :: x(n)
    real(real64),         intent(inout) :: y(n)

    integer, parameter :: chunk = 65536       ! elements handed to one call
    integer            :: c, lo, m

    !$omp parallel do schedule(static) private(lo, m)
    do c = 0, (n - 1)/chunk
       lo = c*chunk + 1
       m  = min(chunk, n - lo + 1)
       call kernel(m, x(lo), y(lo))           ! sequence association: x(lo:lo+m-1)
    end do
    !$omp end parallel do
  end subroutine par_copy

  !> Time one kernel: warm up once, then report the best of niter copies
  !! (the STREAM convention -- the fastest run best reflects the hardware and
  !! is the most reproducible on a noisy, shared machine).
  subroutine run_bench(label, kernel, n, x, y, niter)
    character(*),         intent(in)    :: label
    procedure(copy_kernel)              :: kernel
    integer,              intent(in)    :: n, niter
    real(real64),         intent(in)    :: x(n)
    real(real64),         intent(inout) :: y(n)

    integer(int64) :: c0, c1, rate
    integer        :: it
    real(real64)   :: secs, best, bytes, gbps, err

    call par_copy(kernel, n, x, y)             ! warm-up (also fills y once)

    call system_clock(count_rate=rate)         ! clock resolution (ticks/sec)
    best = huge(1.0_real64)
    do it = 1, niter
       call system_clock(c0)
       call par_copy(kernel, n, x, y)
       call system_clock(c1)
       secs = real(c1 - c0, real64) / real(rate, real64)
       best = min(best, secs)
    end do
    best = max(best, 1.0_real64 / real(rate, real64))   ! guard sub-tick timing

    bytes = 2.0_real64 * real(n, real64) * real(storage_size(x)/8, real64)
    gbps  = bytes / best / 1.0e9_real64
    err   = maxval(abs(y - x))                 ! a pure copy => must be 0

    write(output_unit, '(a,t22,f10.2,a,f9.2,a,es9.1)') &
         trim(label), best * 1.0e6_real64, ' us/copy', gbps, ' GB/s   max|err|=', err
    write(output_unit, '("DATA,",a,",",i0,",",f0.2,",",f0.2,",",es0.1)') &
         trim(label), n, best * 1.0e6_real64, gbps, err
  end subroutine run_bench

end module dcopy_bench_mod


program dcopy_bench
  use dcopy_bench_mod
  implicit none

  integer                   :: n, niter, nargs, ios, i
  character(len=64)         :: arg
  character(len=64)         :: threads
  integer                   :: tlen, tstat
  real(real64), allocatable :: x(:), y(:)
  real(real64)              :: mb_per_vec

  n     = 8000000        ! 64 MB per vector at real64
  niter = 50

  nargs = command_argument_count()
  if (nargs >= 1) then
     call get_command_argument(1, arg)
     if (trim(arg) == '-h' .or. trim(arg) == '--help') then
        call print_help(); stop
     end if
     read(arg, *, iostat=ios) n
     if (ios /= 0 .or. n < 1) then
        write(*,*) 'error: invalid n'; stop 1
     end if
  end if
  if (nargs >= 2) then
     call get_command_argument(2, arg)
     read(arg, *, iostat=ios) niter
     if (ios /= 0 .or. niter < 1) then
        write(*,*) 'error: invalid iterations'; stop 1
     end if
  end if

  allocate(x(n), y(n))
  do i = 1, n
     x(i) = real(i, real64) * 1.0e-3_real64
  end do

  mb_per_vec = real(n, real64) * real(storage_size(x)/8, real64) / 1.0e6_real64  ! decimal MB, matching GB/s

  call get_environment_variable('OMP_NUM_THREADS', threads, tlen, tstat)
  if (tstat /= 0 .or. tlen == 0) threads = '(unset: serial or runtime default)'

  write(*,'(a)')        '------------------------------------------------------------'
  write(*,'(a)')        ' DCOPY: standard stores vs non-temporal stores   y = x'
  write(*,'(a)')        '------------------------------------------------------------'
  write(*,'(a,i0)')     ' elements per vector : ', n
  write(*,'(a,f0.1,a)') ' size per vector     : ', mb_per_vec, ' MB'
  write(*,'(a,i0)')     ' timed copies (best) : ', niter
  write(*,'(a,a)')      ' OMP_NUM_THREADS     : ', trim(threads)
  write(*,'(a)')        ' effective bandwidth counts one read + one write per element'
  write(*,'(a)')        '------------------------------------------------------------'

  ! Prefill y with a sentinel before each run so a partial write shows up in max|err|.
  y = -huge(1.0_real64);  call run_bench('array syntax',       dcopy_array,  n, x, y, niter)
  y = -huge(1.0_real64);  call run_bench('memcpy',             dcopy_memcpy, n, x, y, niter)
  y = -huge(1.0_real64);  call run_bench('standard store',     dcopy_store,  n, x, y, niter)
  y = -huge(1.0_real64);  call run_bench('non-temporal store', dcopy_nt,     n, x, y, niter)

  write(*,'(a)')        '------------------------------------------------------------'
  write(*,'(a)')        ' note: non-temporal stores skip the read-for-ownership traffic'
  write(*,'(a)')        '       and cache pollution that ordinary stores incur, so their'
  write(*,'(a)')        '       edge grows as memory bandwidth becomes the bottleneck --'
  write(*,'(a)')        '       larger vectors and, above all, more threads.  Re-run with'
  write(*,'(a)')        '       OMP_NUM_THREADS=1,2,4,... to watch the gap change.'

  deallocate(x, y)

contains

  subroutine print_help()
    write(*,'(a)') 'Usage: ./dcopy_bench [n] [iterations]'
    write(*,'(a)') '  n            elements per vector (default 8000000)'
    write(*,'(a)') '  iterations   timed copies per variant (default 50)'
  end subroutine print_help

end program dcopy_bench
