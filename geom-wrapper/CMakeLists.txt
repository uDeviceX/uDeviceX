cmake_minimum_required(VERSION 2.8.10)

find_package(CGAL QUIET COMPONENTS Core )
if (CGAL_FOUND)
      include(${CGAL_USE_FILE})
      add_library(geom-wrapper geom-wrapper.cpp geom-wrapper.h rbc_utils.cpp rbc_utils.h ud2f.impl.h)
      set_target_properties(geom-wrapper PROPERTIES LINK_SEARCH_START_STATIC 1)
      set_target_properties(geom-wrapper PROPERTIES LINK_SEARCH_END_STATIC 1)

        # use `auto' in c++
	target_compile_features(geom-wrapper PRIVATE cxx_auto_type)

      add_subdirectory(demo)
  else()
    message(STATUS "This program requires the CGAL library, and will not be compiled.")
endif()
