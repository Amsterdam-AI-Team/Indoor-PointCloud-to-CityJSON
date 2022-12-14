#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>

#include <CGAL/property_map.h>
#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>

#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygonal_surface_reconstruction.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>

#include <utility>
#include <fstream>
#include <CGAL/Timer.h>

#ifdef CGAL_USE_SCIP  // defined (or not) by CMake scripts, do not define by hand
#include <CGAL/SCIP_mixed_integer_program_traits.h>
typedef CGAL::SCIP_mixed_integer_program_traits<double> MIP_Solver;
#elif defined(CGAL_USE_GLPK)  // defined (or not) by CMake scripts, do not define by hand
#include <CGAL/GLPK_mixed_integer_program_traits.h>
typedef CGAL::GLPK_mixed_integer_program_traits<double> MIP_Solver;
#endif

#if defined(CGAL_USE_GLPK) || defined(CGAL_USE_SCIP)

#define MIN_PARAMETERS 11

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT FT;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
// Point with normal, and plane index
typedef boost::tuple<Point, Vector, int> PNI;
typedef std::vector<PNI> Point_vector;
typedef CGAL::Nth_of_tuple_property_map<0, PNI> Point_map;
typedef CGAL::Nth_of_tuple_property_map<1, PNI> Normal_map;
typedef CGAL::Nth_of_tuple_property_map<2, PNI> Plane_index_map;
typedef CGAL::Polygonal_surface_reconstruction<Kernel> Polygonal_surface_reconstruction;
typedef CGAL::Surface_mesh<Point> Surface_mesh;
typedef CGAL::Shape_detection::Efficient_RANSAC_traits<Kernel, Point_vector, Point_map, Normal_map> Traits;
typedef CGAL::Shape_detection::Efficient_RANSAC<Traits> Efficient_ransac;
typedef CGAL::Shape_detection::Plane<Traits> Plane;
typedef CGAL::Shape_detection::Point_to_shape_index_map<Traits> Point_to_shape_index_map;

// Concurrency
typedef CGAL::Parallel_if_available_tag Concurrency_tag;

class Index_map 
{
    public:
        using key_type = std::size_t;
        using value_type = int;
        using reference = value_type;
        using category = boost::readable_property_map_tag;

        Index_map() { }
        template<typename PointRange> Index_map(const PointRange& points,
                    const std::vector< std::vector<std::size_t> >& regions)
                    : m_indices(new std::vector<int>(points.size(), -1)) {
            for (std::size_t i = 0; i < regions.size(); ++i)
                for (const std::size_t idx : regions[i])
                    (*m_indices)[idx] = static_cast<int>(i);
        }

        inline friend value_type get(const Index_map& index_map,
                                    const key_type key) {
            const auto& indices = *(index_map.m_indices);
            return indices[key];
        }

    private:
        std::shared_ptr< std::vector<int> > m_indices;
};

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
  float probability = std::stof(argv[3]);
  float min_points = std::stof(argv[4]);
  float epsilon = std::stof(argv[5]);
  float cluster_epsilon = std::stof(argv[6]);
  float normal_threshold = std::stof(argv[7]);
  float fitting = std::stof(argv[8]);
  float coverage = std::stof(argv[9]);
  float complexity = std::stof(argv[10]);

  if ((fitting + coverage + complexity) >  1.0f) {
    std::cout << fitting << " " << coverage << " " << complexity << std::endl;
    std::cerr << "Parameters sum to greater than 1" << std::endl;
    return EXIT_FAILURE; 
  };

  Point_vector points;

  // Load point set from a file.
  const std::string& input_file(input_file_name);
  std::ifstream input_stream(input_file.c_str(), std::ios::binary);
  std::cout << "Loading point cloud: " << input_file << "...";
  CGAL::Timer t;
  t.start();

  if (!CGAL::IO::read_PLY(input_stream,
                      std::back_inserter(points),
                      CGAL::parameters::point_map(Point_map()))) {
      std::cerr << "Error: cannot read file " << input_file << std::endl;
      return EXIT_FAILURE;
  }
  else
      std::cout << " Done. " << points.size() << " points. Time: "
      << t.time() << " sec." << std::endl;

  std::cout << "Estimating normals...";
  t.reset();

  // // Radius
  // CGAL::pca_estimate_normals<Concurrency_tag>(points,
  //                   16, // limit on the number of neighbors
  //                   CGAL::parameters::point_map(Point_map())
  //                                   .normal_map(Normal_map())
  //                                   .neighbor_radius(0.10));

  // Radius
  CGAL::pca_estimate_normals<Concurrency_tag>(points,
                    16, // limit on the number of neighbors
                    CGAL::parameters::point_map(Point_map())
                                    .normal_map(Normal_map()));

  std::cout << " Done. Time: " << t.time() << " sec." << std::endl;

  //////////////////////////////////////////////////////////////////////////

  // Shape detection.

  // Set parameters for shape detection.
  Efficient_ransac::Parameters parameters;    
  parameters.probability = probability; // Probability to miss the largest primitive at each iteration.
  parameters.min_points = min_points; // Detect shapes with at least number points.
  parameters.epsilon = epsilon; // Max distance between a point and a shape.
  parameters.cluster_epsilon = cluster_epsilon; // Maximum distance between points to be clustered.
  parameters.normal_threshold = normal_threshold; // Mximum normal deviation. 0.9 < dot(surface_normal, point_normal);

  Efficient_ransac ransac;
  ransac.set_input(points);
  ransac.add_shape_factory<Plane>();

  std::cout << "Extracting planes...";
  t.reset();
  ransac.detect(parameters);

  Efficient_ransac::Plane_range planes = ransac.planes();
  std::size_t num_planes = planes.size();

  std::cout << " Done. " << planes.size() << " planes extracted. Time: " 
  << t.time() << " sec." << std::endl;

  // Print number of detected shapes and unassigned points.
  std::cout << ransac.shapes().end() - ransac.shapes().begin()
  << " detected shapes, "
  << ransac.number_of_unassigned_points()
  << " unassigned points." << std::endl;

  // Stores the plane index of each point as the third element of the tuple.
  Point_to_shape_index_map shape_index_map(points, planes);
  for (std::size_t i = 0; i < points.size(); ++i) {
      // Uses the get function from the property map that accesses the 3rd element of the tuple.
      int plane_index = get(shape_index_map, i);
      points[i].get<2>() = plane_index;
  }

  //////////////////////////////////////////////////////////////////////////

  // Write point set
  const std::string& output_file_ply(output_path + "/ransac_result.ply");
  std::ofstream output_stream_ply(output_file_ply.c_str(), std::ios_base::binary);
  CGAL::IO::set_binary_mode(output_stream_ply); // The PLY file will be written in the binary format
  if (CGAL::IO::write_PLY_with_properties(output_stream_ply,
                                points,
                                CGAL::make_ply_point_writer (Point_map()),
                                CGAL::make_ply_normal_writer (Normal_map()),
                                std::make_pair(Plane_index_map(), CGAL::IO::PLY_property<int>("plane_index")))) {
      std::cout << " Done. Saved to " << output_file_ply << std::endl;
  }
  else {
      std::cerr << " Failed saving file." << std::endl;
      return EXIT_FAILURE;
  }

  //////////////////////////////////////////////////////////////////////////

  // Polygonal Surface Reconstruction

  Polygonal_surface_reconstruction algo(
    points,
    Point_map(),
    Normal_map(),
    Plane_index_map()
  );

  Surface_mesh model;

  std::cout << "Polyfit...";
  t.reset();

  if (!algo.reconstruct<MIP_Solver>(model, fitting, coverage, complexity)) {
    std::cerr << " Failed: " << algo.error_message() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << " Done. Time: " << t.time() << " sec." << std::endl;

  if (model.is_empty()) {
    std::cerr << " Failed: no vertices" << std::endl;
    return EXIT_FAILURE;
  }

  // Saves the mesh model
  const std::string& output_file_mesh(output_path + "/polyfit_result.obj");
  std::ofstream output_stream_mesh(output_file_mesh.c_str());
  if (output_stream_mesh && CGAL::IO::write_OBJ(output_stream_mesh, model))
    std::cout << " Done. Saved to " << output_file_mesh << ". Time: " << t.time() << " sec." << std::endl;
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