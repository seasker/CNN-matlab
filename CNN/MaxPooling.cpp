/*
 * MaxPooling.c
 *
 * Implements the max-pooling transfer function. Takes a 4D tensor shaped as
 * (rows, cols, nchannels, nsamples) and a pooling shape as (prows, pcols) and
 * returns a set of max-values with the corresponding indices in the input
 * matrix.
 *
 * e.g.
 *      [m, idx] = MaxPooling(IM, [2 2])
 *
 *  Created on: July 11, 2011
 *      Author: Jonathan Masci <jonathan@idsia.ch>
 * 
 * This file is available under the terms of the GNU GPLv2.
 */

#include "mex.h"

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int debug = 0;

/**
 * Computes the max-pooling for the given 2D map, and no, the name is not a typo.
 * All pointers are passed already offset so to avoid cumbersome indexing.
 *
 * @param ptr_data pointer set to the begin of this map
 * @param DATA_DIMS data dimensions
 * @param ptr_pool pooling sizes
 * @param ptr_out pointer to the output max-values set to the right position
 * @param ptr_idx pointer to the outpumxClassID classID)t indices set to the right position
 */
//  compute_map_pooling (&ptr_data[data_offset], DATA_DIMS, ptr_pool, &ptr_out[out_offset], &ptr_idx[out_offset], data_offset);
template <typename T>
inline void compute_map_pooling(T *ptr_data, const mwSize *DATA_DIMS, T *ptr_poolsize, T *ptr_poolstride,
    T *ptr_out, T *ptr_idx, int data_offset)
{
  T m;
  int idx;
  int count = 0;
  //FILE *fp = fopen("zz","a+");
  //fprintf(fp,"data_offset=%d, poolsize[%f,%f], poolstride[%f,%f]\n", data_offset, ptr_poolsize[0],ptr_poolsize[1],ptr_poolstride[0], ptr_poolstride[1]);
  for (int col = 0; col <= DATA_DIMS[1]-ptr_poolsize[1]; col += ptr_poolstride[1]) {
      for (int row = 0; row <= DATA_DIMS[0]-ptr_poolsize[0]; row += ptr_poolstride[0]) {
          if (debug)
            fprintf(stderr, "r = %i,c = %i \n", row, col);
         
          m = -std::numeric_limits<T>::max();
          idx = -1;
          for (int pcol = 0; (pcol < ptr_poolsize[1] && col + pcol < DATA_DIMS[1]); ++pcol) {
              for (int prow = 0; (prow < ptr_poolsize[0] && row + prow < DATA_DIMS[0]); ++prow) {
                  if (debug) {
                      fprintf(stderr, "m = %f, data = %f \n", m, ptr_data[IDX2C(row + prow, col + pcol, DATA_DIMS[0])]);
                      fprintf(stderr, "rr = %i, cc = %i \n --> idx = %i \n", row + prow, col + pcol, idx);
                  }

                  if (ptr_data[IDX2C(row + prow, col + pcol, DATA_DIMS[0])] > m) {
                      idx = IDX2C(row + prow, col + pcol, DATA_DIMS[0]);
                      m = ptr_data[idx];
                  }
              }//for
          }//for
          
          if (debug && idx == -1) {
              fprintf(stderr, "dimension overflow or data has NaN or so\n");
              return;
          }

            //fprintf(fp, "count = %i,Datadim[%d,%d]\n",count, DATA_DIMS[0], DATA_DIMS[1]);

          /* idxs are to be used in Matlab and hence a +1 is needed */
          ptr_idx[count] = idx + 1 + data_offset;
          ptr_out[count] = m;
          count++;
      }//for row
  }//for col

 //fclose(fp);
}
/**
 * This is the wrapper for the actual computation.
 * It is a template so that multiple types can be handled.
 */
template <typename T>
void mexMaxPooling(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], mxClassID classID)
{

  /***************************************************************************/
  /** Variables */
  /***************************************************************************/
  mwSize IDX_DIMS[1];
  mwSize DATA_DIMS[4];
  mwSize M_DIMS[4];
  const mwSize *POOL_DIMS;
  int DATA_NUMEL;
  int POOL_NUMEL;

  /**
   * Pointers to data
   */
  T *ptr_data = NULL;
  T *ptr_poolsize = NULL;
  T *ptr_poolstride = NULL;
  T *ptr_out = NULL;
  T *ptr_idx = NULL;

  /***************************************************************************/
  /** Setting input pointers *************************************************/
  /***************************************************************************/
  ptr_data = (T *)mxGetData(prhs[0]);
  ptr_poolsize = (T *)mxGetData(prhs[1]);
  ptr_poolstride = (T *)mxGetData(prhs[2]);
  //FILE * fp = fopen("zz","a+");
  if (debug)
    fprintf(stderr,"Pooling size: poolh=%f, poolw=%f, poolstride=%f\n ", ptr_poolsize[0], ptr_poolsize[1], ptr_poolstride[0]);

// fclose(fp);
  /***************************************************************************/
  /** Setting parameters *****************************************************/
  /***************************************************************************/
  /* Data dimensions. As also a 2D tensor can be used I fill empty dimensions
   * with 1 */
  const mwSize *test = mxGetDimensions(prhs[1]);
  
  const mwSize *tmp = mxGetDimensions(prhs[0]);
  DATA_DIMS[0] = tmp[0];
  DATA_DIMS[1] = tmp[1];

  if (mxGetNumberOfDimensions(prhs[0]) == 2) {
      DATA_DIMS[2] = 1;
      DATA_DIMS[3] = 1;
  } else if (mxGetNumberOfDimensions(prhs[0]) == 3) {
      DATA_DIMS[2] = tmp[2];
      DATA_DIMS[3] = 1;
  } else {
      DATA_DIMS[2] = tmp[2];
      DATA_DIMS[3] = tmp[3];
  }

  DATA_NUMEL = DATA_DIMS[0] * DATA_DIMS[1] * DATA_DIMS[2] * DATA_DIMS[3];
  if (debug)
    fprintf(stderr,"Data size: h=%d, w=%d, z=%d, n=%d (%d)\n", DATA_DIMS[0], DATA_DIMS[1], DATA_DIMS[2], DATA_DIMS[3], DATA_NUMEL);

  /* Output dimensions: the first output argument is of size equals to the input
   * whereas the second is of size equals to the number of pooled values.
   * Below there is ceil because also non complete tiles are considered when
   * input dims are not multiples of pooling dims. */
   //fprintf(fp,"Pooling size: data_dims0=%d, data_dims1=%d, data_dims2=%d, data_dims3=%d\n ", DATA_DIMS[0], DATA_DIMS[1], DATA_DIMS[2],DATA_DIMS[3]);

  M_DIMS[0] = ceil(((float)DATA_DIMS[0]-(float)ptr_poolsize[0])/ ptr_poolstride[0])+1;
  M_DIMS[1] = ceil(((float)DATA_DIMS[1]-(float)ptr_poolsize[1])/ ptr_poolstride[1])+1;
  
  M_DIMS[2] = DATA_DIMS[2];
  M_DIMS[3] = DATA_DIMS[3];
  IDX_DIMS[0] = M_DIMS[0] * M_DIMS[1] * M_DIMS[2] * M_DIMS[3];
  //fprintf(fp,"Pooling size: MDIMS0=%d, MDIMS1=%d, M_DIMS2=%d, M_DIMS3=%d\n ", M_DIMS[0], M_DIMS[1], M_DIMS[2], M_DIMS[3]);
  
  if (debug){
      fprintf(stderr,"Each output image has (%d, %d) pooled values, "
          "IDXs size: h=%d \n", M_DIMS[0], M_DIMS[1], IDX_DIMS[0]);
      fprintf(stderr, "M size: h=%d, w=%d, z=%d, n=%d\n", M_DIMS[0], M_DIMS[1], M_DIMS[2], M_DIMS[3]);
  }

  /***************************************************************************/
  /** Variables allocation ***************************************************/
  /***************************************************************************/
  /* OUTPUTS: max-values and corresponding indices */
  plhs[0] = mxCreateNumericArray(4, M_DIMS, classID, mxREAL);
  ptr_out = (T *)mxGetData(plhs[0]);
  plhs[1] = mxCreateNumericArray(1, IDX_DIMS, classID, mxREAL);
  ptr_idx = (T *)mxGetData(plhs[1]);
 /*ptr_stride = (T *)mxGetData(plhs[2]);
 ptr_stride[0] =*/
  /***************************************************************************/
  /** Compute max-pooling ****************************************************/
  /***************************************************************************/
  int out_offset = 0;
  int data_offset = 0;
  int M_sample_size = M_DIMS[0] * M_DIMS[1] * M_DIMS[2];
  int D_sample_size = DATA_DIMS[0] * DATA_DIMS[1] * DATA_DIMS[2];
  
  //fclose(fp);
  for (int n = 0; n < DATA_DIMS[3]; ++n) {
#pragma omp parallel for
      for (int k = 0; k < DATA_DIMS[2]; ++k) {
          out_offset = n * M_sample_size + k * M_DIMS[0] * M_DIMS[1];
          data_offset = n * D_sample_size + k * DATA_DIMS[0] * DATA_DIMS[1];

           
          compute_map_pooling (&ptr_data[data_offset], DATA_DIMS, ptr_poolsize, ptr_poolstride, 
                  &ptr_out[out_offset], &ptr_idx[out_offset], data_offset);

          if (debug)
            fprintf(stderr, "out_offset: %i, data_offset: %i\n", out_offset, data_offset);
      }//for
  }//for
    //fp = fopen("zz","a+");
    //fprintf(fp, "have done\n");
    //fclose(fp);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /***************************************************************************/
  /** Check input ************************************************************/
  /***************************************************************************/
  if (nrhs !=3)
    mexErrMsgTxt("Must have 3 input arguments: x, poolingsize, poolingstride");

  /*if (nlhs !=2)
    mexErrMsgTxt("Must have 2 output arguments ([max_value, idxs])");*/

  if (mxIsComplex(prhs[0]) || !(mxIsClass(prhs[0],"single") || mxIsClass(prhs[0],"double")))
    mexErrMsgTxt("Input data must be real, single/double type");

  if (mxIsComplex(prhs[1]) || !(mxIsClass(prhs[1],"single") || mxIsClass(prhs[1],"double")))
    mexErrMsgTxt("Pooling size (rows, cols) must be real, single/double type");
  
  if (mxIsComplex(prhs[2]) || !(mxIsClass(prhs[2],"single") || mxIsClass(prhs[2],"double")))
    mexErrMsgTxt("Pooling stride  must be real, single/double type");
  
  if (mxGetNumberOfDimensions(prhs[0]) < 2)
    mexErrMsgTxt("Input data must have at least 2-dimensions (rows, cols, nchannels, nsamples) "
        "\nThe last two dimensions will be considered to be 1.");

  if (mxGetNumberOfDimensions(prhs[1]) != 2)
    mexErrMsgTxt("Pooling size must have 2-dimensions (prows, pcols)");

  mxClassID classID = mxGetClassID(prhs[0]);

  /** This is mainly to avoid two typenames. Should not be a big usability issue. */
  if (mxGetClassID(prhs[1]) != classID || mxGetClassID(prhs[2]) != classID)
    mexErrMsgTxt("Input data and pooling need to be of the same type");

  /***************************************************************************/
  /** Switch for the supported data types */
  /***************************************************************************/
  if (classID == mxSINGLE_CLASS) {
      if (debug)
        fprintf(stderr, "Executing the single version\n");

      mexMaxPooling<float>(nlhs, plhs, nrhs, prhs, classID);
  } 
  else if (classID == mxDOUBLE_CLASS) {
      if (debug)
        fprintf(stderr, "Executing the double version\n");

      mexMaxPooling<double>(nlhs, plhs, nrhs, prhs, classID);
  }
}
