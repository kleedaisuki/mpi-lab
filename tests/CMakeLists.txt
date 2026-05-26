add_executable(mpilab_smoke_test
    smoke/smoke.cpp
)

target_link_libraries(mpilab_smoke_test
    PRIVATE
        mpi_svd_lab::mpilab
)

mpilab_apply_common_target_options(mpilab_smoke_test)

add_test(NAME mpilab.smoke COMMAND mpilab_smoke_test)
