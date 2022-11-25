# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

project(polyfit)

cmake_minimum_required(VERSION 3.1...3.23)

# CGAL and its components
find_package(CGAL REQUIRED)

# Boost and its components
find_package(Boost REQUIRED)

if(NOT Boost_FOUND)

  message(
    STATUS
      "NOTICE: This project requires the Boost library, and will not be compiled."
  )

  return()

endif()

# Creating entries for all C++ files with "main" routine
# ##########################################################

find_package(Eigen3 3.1.0) #(requires 3.1.0 or greater)
include(CGAL_Eigen3_support)
if(NOT TARGET CGAL::Eigen3_support)
  message(
    STATUS
      "NOTICE: This project requires Eigen 3.1 (or greater) and will not be compiled."
  )
  return()
endif()

find_package(GLPK QUIET)
include(CGAL_GLPK_support)
if(NOT TARGET CGAL::GLPK_support)
  message(
    STATUS
      "NOTICE: This project requires either GLPK, and will not be compiled."
  )
  return()
endif()

create_single_source_cgal_program("polyfit.cpp")
target_link_libraries(polyfit PUBLIC CGAL::Eigen3_support)
target_link_libraries(polyfit PUBLIC CGAL::GLPK_support)