#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#define DEG_TO_RAD 0.01745329251
#define EARTH_RADIUS_KM 6371.0088
#define THREADS_PER_BLOCK 1024

namespace py = pybind11;

__global__
void haversine(
		int size,
		const double* x1,
		const double* y1,
		const double* x2,
		const double* y2,
		double* out
)
{
	int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (idx >= size)
	{
		return;
	}

	double phi1 = y1[idx] * DEG_TO_RAD;
	double phi2 = y2[idx] * DEG_TO_RAD;
	double lambda1 = x1[idx] * DEG_TO_RAD;
	double lambda2 = x2[idx] * DEG_TO_RAD;

	double d_phi = phi2 - phi1;
	double d_lambda = lambda2 - lambda1;

	double hav = (1 - cos(d_phi) + cos(phi1) * cos(phi2) * (1 - cos(d_lambda))) / 2;

	out[idx] = 2 * EARTH_RADIUS_KM * asin(sqrt(hav));
}

void distance(
		int size,	
		py::array_t<double> py_x1,
		py::array_t<double> py_y1,
		py::array_t<double> py_x2,
		py::array_t<double> py_y2,
		py::array_t<double> py_out
)
{
	py::buffer_info buf_x1 = py_x1.request();
	py::buffer_info buf_y1 = py_y1.request();
	py::buffer_info buf_x2 = py_x2.request();
	py::buffer_info buf_y2 = py_y2.request();
	py::buffer_info buf_out = py_out.request();

	// Make sure that the data is in the form we expect
	assert(buf_x1.ndim == 1 && buf_x1.size == size);
	assert(buf_y1.ndim == 1 && buf_y1.size == size);
	assert(buf_x2.ndim == 1 && buf_x2.size == size);
	assert(buf_y2.ndim == 1 && buf_y2.size == size);
	assert(buf_out.ndim == 1 && buf_out.size == size);

	// Copy the input data to the device
	double* x1 = reinterpret_cast<double*>(buf_x1.ptr);
	double* y1 = reinterpret_cast<double*>(buf_y1.ptr);
	double* x2 = reinterpret_cast<double*>(buf_x2.ptr);
	double* y2 = reinterpret_cast<double*>(buf_y2.ptr);

	thrust::device_vector<double> d_x1(x1, x1 + size);
	thrust::device_vector<double> d_y1(y1, y1 + size);
	thrust::device_vector<double> d_x2(x2, x2 + size);
	thrust::device_vector<double> d_y2(y2, y2 + size);
	thrust::device_vector<double> d_out(size);

	// Run the kernel function
	int blockSize = ceil((float) size / THREADS_PER_BLOCK);

	haversine<<<blockSize, THREADS_PER_BLOCK>>>(
			size,
			thrust::raw_pointer_cast(d_x1.data()),
			thrust::raw_pointer_cast(d_y1.data()),
			thrust::raw_pointer_cast(d_x2.data()),
			thrust::raw_pointer_cast(d_y2.data()),
			thrust::raw_pointer_cast(d_out.data()));

	// Copy the output back to the Python buffer
	double* out = reinterpret_cast<double*>(buf_out.ptr);
	thrust::copy(d_out.begin(), d_out.end(), out);
}

// Define the Python FFI bindings
PYBIND11_MODULE(haversine, m)
{
	m.doc() = "Parallelized implementation of the Haversine distance formula";
	m.def("distance", distance);
}

