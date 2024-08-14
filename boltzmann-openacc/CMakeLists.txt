cmake_minimum_required(VERSION 3.10)
project(boltzmann-openacc)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the OpenACC detection module
include(FindOpenACC)

if(OpenACC_FOUND)
    message(STATUS "OpenACC detected")

    # Set compile options for C and C++
    if(OpenACC_C_FOUND)
        message(STATUS "OpenACC C flags: ${OpenACC_C_FLAGS}")
        add_compile_options(${OpenACC_C_FLAGS})
    endif()

    if(OpenACC_CXX_FOUND)
        message(STATUS "OpenACC CXX flags: ${OpenACC_CXX_FLAGS}")
        add_compile_options(${OpenACC_CXX_FLAGS})
    endif()

else()
    message(FATAL_ERROR "OpenACC not found")
endif()

# Add source files
set(SOURCES
    openacc b2q9_openACC.c 
)

# Create the executable
add_executable(boltzmann-openacc b2q9_openACC.c)

# Link OpenACC libraries if any are found
if(OpenACC_C_FOUND OR OpenACC_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}
        $<$<COMPILE_LANGUAGE:C>:OpenACC::OpenACC_C>
        $<$<COMPILE_LANGUAGE:CXX>:OpenACC::OpenACC_CXX>
    )

endif()
