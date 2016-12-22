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
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include "cs43805351.h"

static const double Delta = 0.005491;
static const double xMid = 0.745796;
static const double yMid = 0.105089;

int main(int argc, char *argv[])
{
  int numNodes,   // Number of processes (p)
      my_rank,    // my process num (0-(p-1))
      my_start,   // starting frame to process 
      my_end;     // last frame to process
 
  // MPI Environment initialization
  MPI_Init(NULL, NULL);
  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD,  &numNodes);
  // Get the rank of this process
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if(my_rank == 0) printf("Fractal v1.5 [MPI]\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "usage: %s frame_width num_frames\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: frame_width must be at least 10\n"); exit(-1);}
  int frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  
  if (my_rank==0)
    printf("computing %d frames of %d by %d fractal\n", frames, width, width);
  
  /* check if frames is a multiple of number of processes */
  if(frames % numNodes != 0) 
  {printf("error: # of frames must be multiple of the # of processes"); exit(-1);}

  // have process 0 print the total number of processes
  if(my_rank == 0) printf("Total number of processes: %d\n", numNodes);

  // amount of frames each processor is responsible for computing
  int frameAmt = frames / numNodes;
  
  // calculating start and end frame indexes
  my_start = my_rank * frameAmt;
  my_end = my_start + frameAmt;

  unsigned char *finalPic = NULL;
  if(my_rank == 0) {
  	// allocate array to hold all frames
   	finalPic = new unsigned char[frames * width * width];
  }
  // smaller array for one process to compute a chunk of frames
  unsigned char* pic = new unsigned char[frameAmt * width * width];
  
  // start time
  struct timeval start, end;
  // call for barrier before timer's started
  MPI_Barrier(MPI_COMM_WORLD); 
  gettimeofday(&start, NULL);

  // each process computes a range/chunk of frames
  double delta = Delta * pow(0.99, my_start);
  for (int frame = my_start; frame < my_end; frame++) {
    delta *= 0.99;
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
        pic[(frame % frameAmt) * width * width + row * width + col] = (unsigned char)depth;
      }
    }
  }

  // GATHER ALL BLOCKS OF FRAMES IN PROCESS 0 using collective communication
  MPI_Gather(pic, frameAmt*width*width, MPI_UNSIGNED_CHAR, finalPic, frameAmt*width*width, 
  	         MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  // end timer
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  if(my_rank == 0) printf("compute time: %.4f s\n", runtime);

  // verify result by writing frames to BMP files
    if ((width <= 400) && (frames <= 30) && (my_rank==0) {
      for (int frame = 0; frame < frames; frame++) {
        char name[32];
          sprintf(name, "fractal%d.bmp", frame + 1000);
        writeBMP(width, width, &finalPic[frame * width * width], name);
      }
    }
  }

  // deallocate memory 
  delete [] pic;
  if(my_rank==0) delete [] finalPic;
  MPI_Finalize();
  return 0;
}
