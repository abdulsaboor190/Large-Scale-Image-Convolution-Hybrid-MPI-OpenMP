/*
 * hello_hybrid.c - toolchain smoke test for MPI + OpenMP.
 *
 * Run:  OMP_NUM_THREADS=2 mpirun -np 2 ./build/hello_hybrid
 * Every MPI rank should report its host and the number of OpenMP threads it
 * actually got. This also checks the thread-support level, which matters in
 * Phase 4 (MPI_THREAD_FUNNELED is enough when only the master thread calls MPI).
 */
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[256];
    gethostname(host, sizeof host);

    int nthreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    if (rank == 0)
        printf("MPI thread level provided: %d (FUNNELED=%d)\n", provided, MPI_THREAD_FUNNELED);

    double t0 = MPI_Wtime(), t1 = omp_get_wtime();
    (void)t0; (void)t1;

    printf("rank %d/%d on %s: %d OpenMP thread(s)\n", rank, size, host, nthreads);

    MPI_Finalize();
    return provided >= MPI_THREAD_FUNNELED ? 0 : 1;
}
