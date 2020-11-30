#include <iostream>
#include <vector>
#include <cmath>
#include <png++/png.hpp>
#include <chrono>
#include <openmpi//mpi.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

double * getGaussian(int height, int width, double sigma, double *kernel) {
    double sum = 0.0;
    for (int i = 0; i < height * width; i++) {
        kernel[i] = exp(-(i / width * i / width + i % width * i % width) / (2 * sigma * sigma)) /
                    (2 * M_PI * sigma * sigma);
        sum += kernel[i];
    }
    for (int i = 0; i < height * width; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

double * loadImage(const char *filename, int *image_width, int *image_height) {
    png::image<png::rgb_pixel> image(filename);
    *image_width = image.get_width();
    *image_height = image.get_height();
    double *imageMatrix = static_cast<double *>(calloc((*image_height) * (*image_width) * 3, sizeof(double)));
    for (int h = 0; h < *image_height; h++) {
        for (int w = 0; w < *image_width; w++) {
            imageMatrix[(h * (*image_width * 3)) + (w * 3)] = image[h][w].red;
            imageMatrix[(h * (*image_width * 3)) + (w * 3) + 1] = image[h][w].green;
            imageMatrix[(h * (*image_width * 3)) + (w * 3) + 2] = image[h][w].blue;
        }
    }
    return imageMatrix;
}

void saveImage(double *result, int height, int width, const char *filename) {
    png::image<png::rgb_pixel> imageFile(width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imageFile[y][x].red = std::max(std::min(result[(y * width * 3) + (x * 3)], 255.0), 0.0);
            imageFile[y][x].green = std::max(std::min(result[(y * width * 3) + (x * 3) + 1], 255.0), 0.0);
            imageFile[y][x].blue = std::max(std::min(result[(y * width * 3) + (x * 3) + 2], 255.0), 0.0);
        }
    }
    imageFile.write(filename);
}

void applyFilter(double *image, int image_width, double *filter, int filter_dimension_size, int start_y,
                 int end_y, double *test, int threads) {
    int height = end_y;
    int newImageHeight = height - filter_dimension_size + 1;
    int newImageWidth = image_width - filter_dimension_size + 1;
    int i, j, h, w;

#pragma omp parallel for default (none) \
    num_threads(threads) \
    private ( h, w, j, i) \
    shared(newImageWidth, newImageHeight, filter_dimension_size, filter, image, start_y, test, image_width)
    for (i = start_y; i < newImageHeight; i++) {
        for (j = 0; j < newImageWidth; j++) {
            for (h = i; h < i + filter_dimension_size; h++) {
                for (w = j; w < j + filter_dimension_size; w++) {
                    test[((i - start_y) * newImageWidth * 3) + (j * 3)] +=
                            filter[((h - i) * filter_dimension_size) + w - j] * image[(h * image_width * 3) + (w * 3)];

                    test[((i - start_y) * newImageWidth * 3) + (j * 3) + 1] +=
                            filter[((h - i) * filter_dimension_size) + w - j] *
                            image[(h * image_width * 3) + (w * 3) + 1];

                    test[((i - start_y) * newImageWidth * 3) + (j * 3) + 2] +=
                            filter[((h - i) * filter_dimension_size) + w - j] *
                            image[(h * image_width * 3) + (w * 3) + 2];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int threads = std::stoi(argv[1]);
    int filter_size = std::stoi(argv[2]);
    int rank, nProcesses, image_width, image_height;
    double *filter = static_cast<double *>(calloc(filter_size * filter_size, sizeof(double)));
    double *result;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

//    Image image = loadImage("test.png");
    double *image = loadImage("test.png", &image_width, &image_height);
    int chunk_y = (image_height - filter_size + 1) / nProcesses;
    double *output_chunk = static_cast<double *>(
            calloc(chunk_y * (image_width - filter_size + 1) * 3, sizeof(double)));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if (rank == 0) {
        std::cout << "\n============IMAGE FILTERv0.1=============\n" <<
                  "Proces count => " << nProcesses <<
                  "\nThread count => " << threads <<
                  "\nFilter size => " << filter_size << "x" << filter_size;
        std::cout << "\n----------------WORK TIME----------------\n";
        result = static_cast<double *>(calloc(chunk_y * (image_width - filter_size + 1) * 3 * nProcesses,
                                              sizeof(double)));
        getGaussian(filter_size, filter_size, 10.0, filter);
    }

    MPI_Bcast(filter, filter_size * filter_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    applyFilter(image,
                image_width,
                filter,
                filter_size,
                (chunk_y * rank),
                (chunk_y * rank) + chunk_y + filter_size - 1,
                output_chunk,
                threads);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "CORE:" << rank << "\tTime difference = "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << "[s]\t("
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << "[ms])" << std::endl;

    MPI_Gather(output_chunk,
               chunk_y * (image_width - filter_size + 1) * 3,
               MPI_DOUBLE,
               result, chunk_y * (image_width - filter_size + 1) * 3,
               MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        saveImage(result, chunk_y * nProcesses, image_width - filter_size + 1, "newImage.png");
        std::cout << "==============>IMAGE SAVED==============> " <<
                  std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count()
                  << "[s]\t" <<
                  std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - begin).count()
                  << "[ms]" << std::endl;;
    }
    MPI_Finalize();
    exit(0);
}