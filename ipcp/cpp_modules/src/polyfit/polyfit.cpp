#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/property_map.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygonal_surface_reconstruction.h>

#ifdef CGAL_USE_SCIP  // defined (or not) by CMake scripts, do not define by hand
#include <CGAL/SCIP_mixed_integer_program_traits.h>
typedef CGAL::SCIP_mixed_integer_program_traits<double>                        MIP_Solver;
#elif defined(CGAL_USE_GLPK)  // defined (or not) by CMake scripts, do not define by hand
#include <CGAL/GLPK_mixed_integer_program_traits.h>
typedef CGAL::GLPK_mixed_integer_program_traits<double>                        MIP_Solver;
#endif

#if defined(CGAL_USE_GLPK) || defined(CGAL_USE_SCIP)

#include <CGAL/Timer.h>

#include <fstream>

#define MIN_PARAMETERS 6

typedef CGAL::Exact_predicates_inexact_constructions_kernel                Kernel;
typedef Kernel::Point_3                                                                                        Point;
typedef Kernel::Vector_3                                                                                Vector;
typedef        CGAL::Polygonal_surface_reconstruction<Kernel>                        Polygonal_surface_reconstruction;
typedef CGAL::Surface_mesh<Point>                                                                Surface_mesh;

// Point with normal, and plane index
typedef boost::tuple<Point, Vector, int>                                                PNI;
typedef CGAL::Nth_of_tuple_property_map<0, PNI>                                        Point_map;
typedef CGAL::Nth_of_tuple_property_map<1, PNI>                                        Normal_map;
typedef CGAL::Nth_of_tuple_property_map<2, PNI>                                        Plane_index_map;

/*
* The following example shows the reconstruction using user-provided
* planar segments stored in PLY format. In the PLY format, a property
* named "segment_index" stores the plane index for each point (-1 if
* the point is not assigned to a plane).
*/

int main(int argc, char *argv[])
{

  if (argc < MIN_PARAMETERS)
  {
    std::cout << "no valid input found" << std::endl;
    return (-1);
  }
  else 
  {
    std::cout << "Succes" << std::endl;
  }
  std::string input_file_name = argv[1];
  std::string output_path = argv[2];
  float fitting = std::stof(argv[3]);
  float coverage = std::stof(argv[4]);
  float complexity = std::stof(argv[5]);
  if ((fitting + coverage + complexity) >  1.0f) {
    std::cout << fitting << " " << coverage << " " << complexity << std::endl;
    std::cerr << "Parameters sum to greater than 1" << std::endl;
    return EXIT_FAILURE; 
  };

  const std::string& input_file(input_file_name);
  std::ifstream input_stream(input_file.c_str(), std::ios::binary);

  std::vector<PNI> points; // store points

  std::cout << "Loading point cloud: " << input_file << "...";
  CGAL::Timer t;
  t.start();

  if (!CGAL::IO::read_PLY_with_properties(input_stream,
                                          std::back_inserter(points),
                                          CGAL::make_ply_point_reader(Point_map()),
                                          CGAL::make_ply_normal_reader(Normal_map()),
                                          std::make_pair(Plane_index_map(), CGAL::PLY_property<int>("segment_index"))))
  {
    std::cerr << "Error: cannot read file " << input_file << std::endl;
    return EXIT_FAILURE;
  }
  else
    std::cout << " Done. " << points.size() << " points. Time: " << t.time() << " sec." << std::endl;

  //////////////////////////////////////////////////////////////////////////

  t.reset();

  Polygonal_surface_reconstruction algo(
    points,
    Point_map(),
    Normal_map(),
    Plane_index_map()
  );

  std::cout << " Done. Time: " << t.time() << " sec." << std::endl;

  //////////////////////////////////////////////////////////////////////////

  Surface_mesh model;

  std::cout << "Reconstructing...";
  t.reset();

  if (!algo.reconstruct<MIP_Solver>(model, fitting, coverage, complexity)) {
    std::cerr << " Failed: " << algo.error_message() << std::endl;
    return EXIT_FAILURE;
  }

  if (model.is_empty()) {
    std::cerr << " Failed: no vertices" << std::endl;
    return EXIT_FAILURE;
  }

  // Saves the mesh model
  const std::string& output_file(output_path + "/polyfit_result.obj");
  std::ofstream output_stream(output_file.c_str());
  if (output_stream && CGAL::IO::write_OBJ(output_stream, model))
    std::cout << " Done. Saved to " << output_file << ". Time: " << t.time() << " sec." << std::endl;
  else {
    std::cerr << " Failed saving file." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

#else

int main(int, char**)
{
    std::cerr << "This test requires either GLPK or SCIP.\n";
    return EXIT_SUCCESS;
}

#endif  // defined(CGAL_USE_GLPK) || defined(CGAL_USE_SCIP)