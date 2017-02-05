cmake_minimum_required(VERSION 2.8.10)

find_package(CGAL QUIET COMPONENTS Core )
if (CGAL_FOUND)
      include(${CGAL_USE_FILE})
      add_library(Hello hello_a.cpp hello_b.cpp)
      set_target_properties(Hello PROPERTIES LINK_SEARCH_START_STATIC 1)
      set_target_properties(Hello PROPERTIES LINK_SEARCH_END_STATIC 1)

      add_subdirectory(Demo)
  else()
    message(STATUS "This program requires the CGAL library, and will not be compiled.")
endif()
