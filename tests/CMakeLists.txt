add_executable(mpilab_smoke_test
    smoke/smoke.cpp
)

target_link_libraries(mpilab_smoke_test
    PRIVATE
        mpi_svd_lab::mpilab
)

mpilab_apply_common_target_options(mpilab_smoke_test)

add_test(NAME mpilab.smoke COMMAND mpilab_smoke_test)

add_executable(mpilab_infrastructure_logger_test
    infrastructure/logger.cpp
)

target_link_libraries(mpilab_infrastructure_logger_test
    PRIVATE
        mpi_svd_lab::mpilab
)

mpilab_apply_common_target_options(mpilab_infrastructure_logger_test)

add_test(NAME mpilab.infrastructure.logger COMMAND mpilab_infrastructure_logger_test)

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

add_executable(mpilab_domain_one_sided_jacobi_svd_test
    domain/one_sided_jacobi_svd.cpp
)

target_link_libraries(mpilab_domain_one_sided_jacobi_svd_test
    PRIVATE
        mpi_svd_lab::mpilab
)

mpilab_apply_common_target_options(mpilab_domain_one_sided_jacobi_svd_test)

add_test(NAME mpilab.domain.one_sided_jacobi_svd COMMAND mpilab_domain_one_sided_jacobi_svd_test)
