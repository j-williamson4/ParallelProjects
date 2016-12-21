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

static const double Delta = 0.005491;
static const double xMid = 0.745796;
static const double yMid = 0.105089;

unsigned char* GPU_Init(const int size);
void GPU_Exec(const int gpu_frames, const int width, unsigned char pic_d[]);
void GPU_Fini(const int size, unsigned char pic[], unsigned char pic_d[]);

int main(int argc, char *argv[])
{
  printf("Fractal v1.5 [Hybrid1]\n");

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
  printf("computing %d frames of %d by %d fractal (%d CPU frames and %d GPU frames)\n", frames, width, width, cpu_frames, gpu_frames);

  // allocate picture arrays
  unsigned char* pic = new unsigned char[frames * width * width];
  unsigned char* pic_d = GPU_Init(gpu_frames * width * width * sizeof(unsigned char));

  // start time
  struct timeval start, end;
  gettimeofday(&start, NULL);

  // the following call should asynchronously compute the given number of frames on the GPU
  GPU_Exec(gpu_frames, width, pic_d);

  // the following code should compute the remaining frames on the CPU

/* insert an OpenMP parallelized FOR loop with 16 threads, default(none), and a cyclic schedule */
  double delta;
  #pragma omp parallel for num_threads(16) default(none) \
  shared(gpu_frames,cpu_frames,width,pic) private(delta) schedule(static,1)
    for (int frame = gpu_frames; frame < (gpu_frames+cpu_frames); frame++) {
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
        pic[frame * width * width + row * width + col] = (unsigned char)depth;
      }
    }
  }
  // the following call should copy the GPU's result into the beginning of the CPU's pic array
  GPU_Fini(gpu_frames * width * width * sizeof(unsigned char), pic, pic_d);

  // end time
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  // verify result by writing frames to BMP files
  if ((width <= 400) && (frames <= 30)) {
    for (int frame = 0; frame < frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 10000);
      writeBMP(width, width, &pic[frame * width * width], name);
    }
  }

  delete [] pic;
  return 0;
}

