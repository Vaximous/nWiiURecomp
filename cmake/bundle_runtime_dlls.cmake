# Copies every non-system DLL an executable needs (recursively) next to it, so
# the build runs from Explorer without the toolchain (e.g. MSYS2) on PATH.
#
#   cmake -DEXE=<path.exe> -DSEARCH_DIRS=<dir;dir> -DOBJDUMP=<objdump> -P bundle_runtime_dlls.cmake
if(NOT EXE OR NOT EXISTS "${EXE}")
    message(FATAL_ERROR "bundle_runtime_dlls: EXE not found: '${EXE}'")
endif()
set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM "windows+pe")
if(OBJDUMP)
    set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL "objdump")
    set(CMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND "${OBJDUMP}")
endif()
file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${EXE}"
    RESOLVED_DEPENDENCIES_VAR resolved
    UNRESOLVED_DEPENDENCIES_VAR unresolved
    DIRECTORIES ${SEARCH_DIRS}
    PRE_EXCLUDE_REGEXES "^api-ms-.*" "^ext-ms-.*"
    POST_EXCLUDE_REGEXES ".*[/\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\].*")
get_filename_component(destination "${EXE}" DIRECTORY)
foreach(dll IN LISTS resolved)
    file(COPY "${dll}" DESTINATION "${destination}")
endforeach()
list(LENGTH resolved count)
message(STATUS "bundled ${count} runtime DLLs next to ${EXE}")
foreach(name IN LISTS unresolved)
    message(STATUS "  not found (expected from the system): ${name}")
endforeach()
