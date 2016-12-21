/*
Fractal code for CS 4380 / CS 5351

Copyright (c) 2016, Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is not permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <sys/time.h>
#include "cs43805351.h"
#include <math.h>
#include <mpi.h>

static const double Delta = 0.005491;
static const double xMid = 0.745796;
static const double yMid = 0.105089;

unsigned char* GPU_Init(const int size);
void GPU_Exec(const int from_frame, const int to_frame, const int width, unsigned char pic_d[]);
void GPU_Fini(const int size, unsigned char pic[], unsigned char pic_d[]);

int main(int argc, char *argv[])
{
  int numNodes, my_rank;

  // MPI Environment initialization
  MPI_Init(NULL, NULL);
  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD,  &numNodes);
  // Get the rank of this process
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if(my_rank == 0) printf("Fractal v1.5 [Hybrid2]\n");
 
  // check command line
  if (argc != 4) {fprintf(stderr, "usage: %s frame_width cpu_frames gpu_frames\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: frame_width must be at least 10\n"); exit(-1);}
  int cpu_frames = atoi(argv[2]);
  if (cpu_frames < 0) {fprintf(stderr, "error: cpu_frames must be at least 0\n"); exit(-1);}
  int gpu_frames = atoi(argv[3]);
  if (gpu_frames < 0) {fprintf(stderr, "error: gpu_frames must be at least 0\n"); exit(-1);}
  int frames = cpu_frames + gpu_frames;
  if (frames < 1) {fprintf(stderr, "error: total number of frames must be at least 1\n"); exit(-1);}
  if(my_rank==0) printf("computing %d frames of %d by %d fractal (%d CPU frames and %d GPU frames)\n", frames*numNodes, width, width, numNodes*cpu_frames, numNodes*gpu_frames);

  const int from_frame = my_rank * frames;
  const int mid_frame = from_frame + gpu_frames;
  const int to_frame = mid_frame + cpu_frames;

  // allocate picture arrays
  unsigned char *finalPic = NULL;
  if(my_rank==0) {
     finalPic = new unsigned char[(frames * numNodes) * width * width];
  }  

  unsigned char* my_pic = new unsigned char[frames* width * width];
  unsigned char* pic_d = GPU_Init((mid_frame-from_frame) * width * width * sizeof(unsigned char));

  // start time
  struct timeval start, end;
  //MPI Barrier before timer
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&start, NULL);

  // the following call should asynchronously compute the given number of frames on the GPU
  GPU_Exec(from_frame, mid_frame, width, pic_d);

  // the following code should compute the remaining frames on the CPU

/* insert an OpenMP parallelized FOR loop with 16 threads, default(none), and a cyclic schedule */
  double delta;
  #pragma omp parallel for num_threads(16) default(none) \
  shared(width,my_pic,frames,numNodes) private(delta) schedule(static,1)
    for (int frame = mid_frame; frame < to_frame; frame++) {
    delta = (Delta * 0.99) * pow(0.99,frame);
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    for (int row = 0; row < width; row++) {
      const double cy = -yMin - row * dw;
      for (int col = 0; col < width; col++) {
        const double cx = -xMin - col * dw;
        double x = cx;
        double y = cy;
        int depth = 256;
        double x2, y2;
        do {
          x2 = x * x;
          y2 = y * y;
          y = 2 * x * y + cy;
          x = x2 - y2 + cx;
          depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));
        my_pic[(frame%(frames/numNodes))* width * width + row * width + col] = (unsigned char)depth;
      }
    }
  }

  // the following call should copy the GPU's result into the beginning of the CPU's pic array
  GPU_Fini((mid_frame-from_frame) * width * width * sizeof(unsigned char), my_pic, pic_d);

  // GATHER ALL BLOCKS OF FRAMES IN PROCESS 0 using collective communication
  MPI_Gather(my_pic, frames*width*width, MPI_UNSIGNED_CHAR, finalPic, frames*width*width, 
  	         MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD); 
  
  // end time
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  if(my_rank == 0) printf("compute time: %.4f s\n", runtime);

  // verify result by writing frames to BMP files
  if ((width <= 400) && (frames <= 30) && (my_rank==0)) {
    for (int frame = 0; frame < frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 10000);
      writeBMP(width, width, &finalPic[frame * width * width], name);
    }
  }

  delete [] my_pic;
  if (my_rank==0) delete [] finalPic;
  MPI_Finalize();
  return 0;
}
