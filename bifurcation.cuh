#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>

#include "kishivi_cuda_tools.cuh"

namespace KiShiVi
{
	namespace CUDA
	{
		__global__ void roSolve(float* paramValues,
								int			nPts,
								int			TMax,
								float		h,
								float		initialCondition1,
								float		initialCondition2,
								float		initialCondition3,
								int			nValue,
								float		prePeakFinderSliceK,
								float*		data,
								int*		dataSizes,
								float		inA,
								float		inB,
								float		inC,
								int			mode);

		void bifurcation(int				_tMax,
						 int				_nPts,
						 float				_h,
						 float				_initialCondition1,
						 float				_initialCondition2,
						 float				_initialCondition3,
						 float				_paramValues1,
						 float				_paramValues2,
						 int				_nValue,
						 float				_prePeakFinderSliceK,
						 float				_paramA,
						 float				_paramB,
						 float				_paramC,
						 int				_mode,
						 float				_memoryLimit,
						 std::string		_outPath,
						 bool				_debug)
		{
			size_t ySize = _tMax / _h;

			size_t free;
			size_t total;

			KiShiVi::CUDA::checkCudaError(cudaMemGetInfo(&free, &total), "cudaMemGetInfo error", _debug);

			float* globalParamValues = linspace(_paramValues1, _paramValues2, _nPts);

			free *= _memoryLimit; // Во избежание приколов будем использовать только 80% от возможной памяти видеокарты
			int maxNpts = free / (sizeof(float) * ((_tMax / _h) + 3)); // Максимально возможное кол-во точек, для обработки в одном потоке

			float*	h_data;
			int*	h_dataSizes;
			float*	h_dataTimes;

			float*	d_data;
			int*	d_dataSizes;
			float*	d_dataTimes;

			if (maxNpts == 0)
			{
				if (_debug)
					std::cout << "\nVery low memory size. Increase the MEMORY_LIMIT!" << "\n";
				exit(1);
			}

			int lastIteration = (_nPts / maxNpts) + 1;

			std::ofstream out;
			out.open(_outPath);

			if(_debug)
				std::cout << "\nCount of memory transfer itarations: " << lastIteration << "\n\n";

			//if (_debug)
			//{
			//	std::cout << "\nProgress:\n";
			//	for (int m = 0; m < 100; ++m)
			//		std::cout << '-';
			//	std::cout << '\n';
			//}

			for (int i = 0; i < lastIteration; ++i)
			{
				if (i == lastIteration - 1)
				{
					h_dataTimes = slice(globalParamValues, maxNpts * i, _nPts);
					maxNpts = _nPts - (maxNpts * i);
				}
				else
				{
					h_dataTimes = slice(globalParamValues, maxNpts * i, maxNpts * i + maxNpts);
				}


				h_data = (float*)malloc(ySize * maxNpts * sizeof(float));
				KiShiVi::CUDA::checkCpuError(h_data, "\nmalloc h_data error", _debug);

				h_dataSizes = (int*)malloc(maxNpts * sizeof(int));
				KiShiVi::CUDA::checkCpuError(h_dataSizes, "\nmalloc h_dataSizes error", _debug);


				KiShiVi::CUDA::checkCudaError(cudaMalloc((void**)&d_data,		ySize * maxNpts * sizeof(float)),	"\ncudaMalloc d_data error",		_debug);
				KiShiVi::CUDA::checkCudaError(cudaMalloc((void**)&d_dataSizes,	maxNpts * sizeof(int)),				"\ncudaMalloc d_dataSizes error", _debug);
				KiShiVi::CUDA::checkCudaError(cudaMalloc((void**)&d_dataTimes,	maxNpts * sizeof(float)),			"\ncudaMalloc d_dataTimes error", _debug);

				KiShiVi::CUDA::checkCudaError(cudaMemcpy(d_dataTimes, h_dataTimes, maxNpts * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice), "\ncudaMemcpy to d_dataTimes error", _debug);


				int blockSize;
				int minGridSize;
				int gridSize;

				cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roSolve, 0, maxNpts);
				gridSize = (maxNpts + blockSize - 1) / blockSize;

				roSolve<<<gridSize, blockSize>>>(d_dataTimes, 
												 maxNpts, 
												 _tMax, 
												 _h, 
												 _initialCondition1, 
												 _initialCondition2,
												 _initialCondition3,
												 _nValue,
												 _prePeakFinderSliceK,
												 d_data,
												 d_dataSizes,
												 _paramA,
												 _paramB,
												 _paramC,
												 _mode);

				KiShiVi::CUDA::checkCudaError(cudaMemcpy(h_data,		d_data,			ySize * maxNpts * sizeof(float),	cudaMemcpyKind::cudaMemcpyDeviceToHost), "cudaMemcpy to h_data error",		_debug);
				KiShiVi::CUDA::checkCudaError(cudaMemcpy(h_dataSizes,	d_dataSizes,	maxNpts * sizeof(float),			cudaMemcpyKind::cudaMemcpyDeviceToHost), "cudaMemcpy to h_dataSizes error", _debug);

				cudaDeviceSynchronize();

				cudaFree(d_data);
				cudaFree(d_dataSizes);
				cudaFree(d_dataTimes);

				for (unsigned int i = 0; i < maxNpts; ++i)
				{
					//std::cout << "#" << i << std::endl;
					for (unsigned int j = 0; j < h_dataSizes[i]; ++j)
					{
						if (out.is_open())
						{
							//std::cout << dataTimes[i] << ", " << data[i * YSize + j] << std::endl;
							out << h_dataTimes[i] << ", " << h_data[i * ySize + j] << '\n';
						}
						else
						{
							std::cout << "\nOutput file open error" << std::endl;
							exit(1);
						}
					}
				}

				std::free(h_data);
				std::free(h_dataSizes);
				std::free(h_dataTimes);

				if (_debug)
				{
					/*for (int m = 0; m < (100 / lastIteration); ++m)
						std::cout << '*';*/
					std::cout << "       " << std::setprecision(3) << (100.0f / (float)lastIteration) * (i + 1) << "%\n";
				}

			}
			if (_debug)
			{
				//for (int m = 0; m < 100 - ((100 / lastIteration) * lastIteration); ++m)
				//	std::cout << '*';
				//std::cout << '\n';
				if (lastIteration != 1)
					std::cout << "       " << "100%\n";
				std::cout << '\n';
			}
			out.close();
		}

		__global__ void roSolve(float*		paramValues,
								int			nPts,
								int			TMax,
								float		h,
								float		initialCondition1,
								float		initialCondition2,
								float		initialCondition3,
								int			nValue,
								float		prePeakFinderSliceK,
								float*		data,
								int*		dataSizes,
								float		inA,
								float		inB,
								float		inC,
								int			mode)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= nPts)
				return;
			float s = paramValues[idx];

			/***********************************************/

			size_t nIter = TMax / h;
			float localH1, localH2;
			float x[3]{ initialCondition1, initialCondition2, initialCondition3 };

			if (mode == SYMMETRY_MODE)
			{
				localH1 = h * s;
				localH2 = h * (1 - s);
			}
			else
			{
				localH1 = h * 0.5;
				localH2 = h * (1 - 0.5);
			}

			float a = inA;
			float b = inB;
			float c = inC;

			switch (mode)
			{
			case PARAM_A_MODE:
				a = s;
				break;
			case PARAM_B_MODE:
				b = s;
				break;
			case PARAM_C_MODE:
				c = s;
				break;
			}

			for (unsigned int i = 0; i < nIter; ++i)
			{
				data[idx * nIter + i] = x[nValue]; // Умножаем номер потока(с учетом блока GPU) на размер одного блока даты + индекс

				x[0] = x[0] + localH1 * (-x[1] - x[2]);
				x[1] = (x[1] + localH1 * (x[0])) / (1 - a * localH1);
				x[2] = (x[2] + localH1 * b) / (1 - localH1 * (x[0] - c));

				x[2] = x[2] + localH2 * (b + x[2] * (x[0] - c));
				x[1] = x[1] + localH2 * (x[0] + a * x[1]);
				x[0] = x[0] + localH2 * (-x[1] - x[2]);
			}

			/***********************************************/

			int _outSize = 0;
			for (size_t i = 1 + (prePeakFinderSliceK * (TMax / h)); i < (TMax / h) - 1; ++i)
			{
				if (data[idx * nIter + i] > data[idx * nIter + i - 1] && data[idx * nIter + i] > data[idx * nIter + i + 1])
				{
					data[idx * nIter + _outSize] = data[idx * nIter + i];
					++_outSize;
				}
				else if (data[idx * nIter + i] > data[idx * nIter + i - 1] && data[idx * nIter + i] == data[idx * nIter + i + 1])
				{
					for (size_t k = i; k < (TMax / h) - 1; ++k)
					{
						if (data[idx * nIter + k] < data[idx * nIter + k + 1])
						{
							break;
							i = k;
						}
						if (data[idx * nIter + k] == data[idx * nIter + k + 1])
							continue;
						if (data[idx * nIter + k] > data[idx * nIter + k + 1])
						{
							data[idx * nIter + _outSize] = data[idx * nIter + k];
							++_outSize;
							i = k + 1;
							break;
						}
					}
				}
			}

			dataSizes[idx] = _outSize;
			//dataSizes[idx] = (TMax / h);
		}
	}
}