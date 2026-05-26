add_executable(mpilab_smoke_test
    smoke/smoke.cpp
)

target_link_libraries(mpilab_smoke_test
    PRIVATE
        mpi_svd_lab::mpilab
)

mpilab_apply_common_target_options(mpilab_smoke_test)

add_test(NAME mpilab.smoke COMMAND mpilab_smoke_test)

add_executable(mpilab_domain_concepts_test
    domain/concepts.cpp
)

target_link_libraries(mpilab_domain_concepts_test
    PRIVATE
        mpi_svd_lab::mpilab
)

mpilab_apply_common_target_options(mpilab_domain_concepts_test)

add_test(NAME mpilab.domain.concepts COMMAND mpilab_domain_concepts_test)

add_executable(mpilab_domain_matrix_layouts_test
    domain/matrix_layouts.cpp
)

target_link_libraries(mpilab_domain_matrix_layouts_test
    PRIVATE
        mpi_svd_lab::mpilab
)

mpilab_apply_common_target_options(mpilab_domain_matrix_layouts_test)

add_test(NAME mpilab.domain.matrix_layouts COMMAND mpilab_domain_matrix_layouts_test)
