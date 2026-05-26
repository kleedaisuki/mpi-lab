include_guard(GLOBAL)

function(mpilab_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4
            /permissive-
            /EHsc
            $<$<BOOL:${MPILAB_WARNINGS_AS_ERRORS}>:/WX>
        )
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wshadow
            -Wdouble-promotion
            -Wformat=2
            -Wundef
            $<$<BOOL:${MPILAB_WARNINGS_AS_ERRORS}>:-Werror>
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE
                -Wextra-semi
                -Wimplicit-fallthrough
                -Wnon-virtual-dtor
            )
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target} PRIVATE
                -Wduplicated-branches
                -Wduplicated-cond
                -Wlogical-op
                -Wuseless-cast
            )
        endif()
    endif()
endfunction()
