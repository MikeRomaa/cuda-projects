#include <climits>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/random/uniform_int_distribution.h>

#define THREADS_PER_BLOCK 1024

/**
 * Definition of a point in 2-dimensional cartesian space.
 *
 * Each component is a value from 0 to USHRT_MAX (65535), inclusive.
 */
struct Point2D
{
	ushort x;
	ushort y;
};

/**
 * Functor to generate a random Point2D struct on the device.
 *
 * Coordinates are uniformly distributed on the range [0, USHRT_MAX].
 */
struct generate_rand_point
{
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<ushort> dist;

	__device__
	Point2D operator()(const std::size_t& offset)
	{
		// Since this is in parallel, the rng won't actually update before the next
		// point is generated. To fix this we manually step through the rng by a
		// unique offset.
		rng.discard(offset * 2);

		return {
			.x = dist(rng),
			.y = dist(rng),
		};
	}
};

/**
 * Functor to get the cell index of a Point2D struct.
 * 
 * The cartesian space is split up into a grid of square cells of equal size. The grid is
 * `grid_size` cells tall and `grid_size` cells wide. Cell indices are assigned in row-major
 * order starting with 0.
 *
 * For example,
 *   +---+---+---+
 *   | 0 | 1 | 2 |
 *   +---+---+---+
 *   | 3 | 4 | 5 |
 *   +---+---+---+
 *   | 6 | 7 | 8 |
 *   +---+---+---+
 *
 * @param `grid_size` Width of the grid.
 */
struct get_cell_idx
{
	ushort grid_size;
	ushort cell_size;

	__host__
	get_cell_idx(ushort _grid_size): grid_size(_grid_size), cell_size(USHRT_MAX / grid_size) {}

	__device__
	ushort operator()(const Point2D& point)
	{
		ushort row = point.y / cell_size;
		ushort col = point.x / cell_size;

		return row * grid_size + col;
	}
};

/**
 * Functor to determine if a given start position represents a non-empty cell.
 *
 * The value `SIZE_MAX` is used as a sentinel value to keep track of whether or not a given cell's
 * start position was set or not.
 */
struct is_non_empty
{
	__device__
	bool operator()(const std::size_t& start_pos)
	{
		return start_pos != SIZE_MAX;
	}
};

/**
 * Compute the start position and length of each cell given a vector of cell indices.
 *
 * This is done by comparing each adjacent pair of cell indices. If there is a change in cell index,
 * then we record the position in the array as the start of the next cell. After all of the start
 * positions are recorded, we separately figure out the length of each cell by subtracting the
 * positions.
 *
 * After the kernel is complete, `out_start[n]` denotes the position in `cell_indices` where the
 * n-th cell begins. Likewise, `out_length[n]` denotes the number of points belonging to the n-th
 * cell.
 *
 * @param `num_points` Total number of points.
 * @param `num_cells` Total number of cells in grid.
 * @param `cell_indices` Pointer to input vector of cell indices. Must be of length `num_points`.
 * @param `out_start` Pointer to output vector of start postions.
 * @param `out_length` Pointer to output vector of cell lengths.
 */
__global__
void process_point_indices(
	std::size_t num_points,
	std::size_t num_cells,
	ushort* cell_indices,
	std::size_t* out_start,
	std::size_t* out_length
)
{
	// If we only have less than two elements, our notion of a "pair" falls apart.
	if (num_points == 0)
	{
		return;
	}
	if (num_points == 1)
	{
		out_start[0] = cell_indices[0];
		out_length[0] = 1;
		return;
	}
	
	std::size_t index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (index >= num_points - 1)
	{
		return;
	}

	ushort cell_a = cell_indices[index];
	ushort cell_b = cell_indices[index + 1];

	// The index of the first cell needs to be set manually since it isn't paired up with
        // a previous one.
	if (index == 0)
	{
		out_start[cell_a] = 0;
	}

	// A change in cell index means a new cell! We must record its start position.
	if (cell_a != cell_b)
	{
		out_start[cell_b] = index + 1;
	}

	// Wait for all of the cell start positions to be computed.
	__syncthreads();

	// Similarly to before, the length of the last cell needs to be manually set since it isn't
	// paired up with a following one.
	if (index == num_points - 2)
	{
		out_length[cell_b] = num_points - out_start[cell_b];
	}

	// This time, a change in cell index means we have to compute the difference.
	if (cell_a != cell_b)
	{
		out_length[cell_a] = out_start[cell_b] - out_start[cell_a];
	}
}

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		std::cerr << "Invalid arguments. "
			<< "Expected usage: ./index_points <num points> <grid size>"
			<< std::endl;

		return EXIT_FAILURE;
	}

	int num_points = std::stoi(argv[1]);
	ushort grid_size = std::stoi(argv[2]);
	ushort num_cells = grid_size * grid_size;

	if (grid_size < 1 || grid_size > 15)
	{
		std::cerr << "Invalid grid size. "
			<< "Expected value between 0 and 15 (inclusive)"
			<< std::endl;

		return EXIT_FAILURE;
	}

	// Randomly generate `num_points` points.

	thrust::device_vector<Point2D> points(num_points);
	thrust::transform(thrust::make_counting_iterator(0),
			  thrust::make_counting_iterator(num_points),
			  points.begin(),
			  generate_rand_point());
	
	// Determine the cell index of each point.

	thrust::device_vector<ushort> cell_indices(num_points);
	thrust::transform(points.begin(),
			  points.end(),
			  cell_indices.begin(),
			  get_cell_idx(grid_size));
	
	// Sort the point vector based on their cell index. This will result in `points` containing
	// all of the points belonging to cell 0 (if any), then all of the points belonging to
	// cell 1 (if any), etc.

	thrust::stable_sort_by_key(cell_indices.begin(), cell_indices.end(), points.begin());

	// Now we use `process_point_indices` to determine the start positions and lengths of each
	// cell in `points`.

	// The vectors are initialized with a sentinel in case a cell has no items.
	// The sentinel is used for filtering in the `is_non_empty` functor.
	thrust::device_vector<std::size_t> start(num_cells, SIZE_MAX);
	thrust::device_vector<std::size_t> length(num_cells, SIZE_MAX);

	// If we have more than 1024 points, we'll need more than one block.
	int block_count = ceil((float) num_points / THREADS_PER_BLOCK);

	process_point_indices<<<block_count, THREADS_PER_BLOCK>>>(
			num_points,
			num_cells,
			thrust::raw_pointer_cast(cell_indices.data()),
			thrust::raw_pointer_cast(start.data()),
			thrust::raw_pointer_cast(length.data()));

	// Next we filter out any empty cells to leave behind just the ones that contain points.
	// We will use the sentinel value from before to determine how many non-empty cells we have.

	int num_nonempty = thrust::count_if(start.begin(), start.end(), is_non_empty());

	thrust::device_vector<ushort> result_cells(num_nonempty);
	thrust::device_vector<std::size_t> result_start(num_nonempty);
	thrust::device_vector<std::size_t> result_length(num_nonempty);

	// Copy the indices of the non-empty cells.
	thrust::copy_if(thrust::make_counting_iterator<ushort>(0),
			thrust::make_counting_iterator<ushort>(num_cells),
			start.begin(),
			result_cells.begin(),
			is_non_empty());

	// Copy the start positions of the non-empty cells.
	thrust::gather(result_cells.begin(),
		       result_cells.end(),
		       start.begin(),
		       result_start.begin());

	// Copy the lengths of the non-empty cells.
	thrust::gather(result_cells.begin(),
		       result_cells.end(),
		       length.begin(),
		       result_length.begin());	

	// Print the result out on the host.

	thrust::host_vector<Point2D> h_points = points;
	thrust::host_vector<ushort> h_result_cells = result_cells;
	thrust::host_vector<std::size_t> h_result_start = result_start;
	thrust::host_vector<std::size_t> h_result_length = result_length;

	printf("%-5s %-10s %-10s\n", "cell", "start", "length");

	for (std::size_t i = 0; i < h_result_cells.size(); i++)
	{
		printf("%-5d %-10ld %-10ld\n",
		       h_result_cells[i],
		       h_result_start[i],
		       h_result_length[i]);
	}

	return EXIT_SUCCESS;
}
