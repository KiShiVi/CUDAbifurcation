#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdio.h>

#include "kishivi_cuda_tools.cuh"

namespace KiShiVi
{
	namespace CUDA
	{
		__global__ void roSolve2d(float* paramValues,
								int			nPts,
								int			TMax,
								float		h,
								float		initialCondition1,
								float		initialCondition2,
								float		initialCondition3,
								int			nValue,
								float		prePeakFinderSliceK,
								float*		data,
								int*		kdeResult,
								float		inA,
								float		inB,
								float		inC,
								int			mode,
								int			kdeSampling,
								float		kdeSamplesInterval1,
								float		kdeSamplesInterval2,
								float		kdeSmoothH);

		void bifurcation2d(int				_tMax,
						 int				_nPts,
						 float				_h,
						 float				_initialCondition1,
						 float				_initialCondition2,
						 float				_initialCondition3,
						 float				_paramValues1,
						 float				_paramValues2,
						 float				_paramValues3,
						 float				_paramValues4,
						 int				_nValue,
						 float				_prePeakFinderSliceK,
						 float				_paramA,
						 float				_paramB,
						 float				_paramC,
						 int				_mode,
						 int				_mode2,
						 float				_memoryLimit,
						 int				_kdeSampling,
						 float				_kdeSamplesInterval1,
						 float				_kdeSamplesInterval2,
						 float				_kdeSmoothH,
						 std::string		_outPath,
						 bool				_debug)
		{
			std::ofstream out;
			out.open(_outPath);

			out << _paramValues1 << ", " << _paramValues2 << "\n" << _paramValues3 << ", " << _paramValues4 << "\n";

			float* global2DParamValues = linspace(_paramValues3, _paramValues4, _nPts);

			float localParamA = _paramA;
			float localParamB = _paramB;
			float localParamC = _paramC;

			for (int i2dIterator = 0; i2dIterator < _nPts; ++i2dIterator)
			{
				switch (_mode2)
				{
				case PARAM_A_MODE:
					localParamA = global2DParamValues[i2dIterator];
					break;
				case PARAM_B_MODE:
					localParamB = global2DParamValues[i2dIterator];
					break;
				case PARAM_C_MODE:
					localParamC = global2DParamValues[i2dIterator];
					break;
				}

				size_t ySize = _tMax / _h;

				size_t free;
				size_t total;

				KiShiVi::CUDA::checkCudaError(cudaMemGetInfo(&free, &total), "cudaMemGetInfo error", _debug);

				float* globalParamValues = linspace(_paramValues1, _paramValues2, _nPts);

				free *= _memoryLimit;
				int maxNpts = free / (sizeof(float) * ((_tMax / _h) + 3)); // Максимально возможное кол-во точек, для обработки в одном потоке

				int* h_kdeResult;
				float* h_dataTimes;

				float* d_data;
				int* d_kdeResult;
				float* d_dataTimes;

				if (maxNpts == 0)
				{
					if (_debug)
						std::cout << "\nVery low memory size. Increase the MEMORY_LIMIT!" << "\n";
					exit(1);
				}

				int lastIteration = (_nPts / maxNpts) + 1;

				//if (_debug)
				//	std::cout << "\nCount of memory transfer itarations: " << lastIteration << "\n\n";

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

					h_kdeResult = (int*)malloc(maxNpts * sizeof(int));
					KiShiVi::CUDA::checkCpuError(h_kdeResult, "\nmalloc h_kdeResult error", _debug);


					KiShiVi::CUDA::checkCudaError(cudaMalloc((void**)&d_data, ySize * maxNpts * sizeof(float)), "\ncudaMalloc d_data error", _debug);
					KiShiVi::CUDA::checkCudaError(cudaMalloc((void**)&d_kdeResult, maxNpts * sizeof(int)), "\ncudaMalloc d_kdeResult error", _debug);
					KiShiVi::CUDA::checkCudaError(cudaMalloc((void**)&d_dataTimes, maxNpts * sizeof(float)), "\ncudaMalloc d_dataTimes error", _debug);

					KiShiVi::CUDA::checkCudaError(cudaMemcpy(d_dataTimes, h_dataTimes, maxNpts * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice), "\ncudaMemcpy to d_dataTimes error", _debug);


					int blockSize;
					int minGridSize;
					int gridSize;

					cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roSolve2d, 0, maxNpts);
					gridSize = (maxNpts + blockSize - 1) / blockSize;

					roSolve2d << <gridSize, blockSize >> > (d_dataTimes,
						maxNpts,
						_tMax,
						_h,
						_initialCondition1,
						_initialCondition2,
						_initialCondition3,
						_nValue,
						_prePeakFinderSliceK,
						d_data,
						d_kdeResult,
						localParamA,
						localParamB,
						localParamC,
						_mode,
						_kdeSampling,
						_kdeSamplesInterval1,
						_kdeSamplesInterval2,
						_kdeSmoothH);

					KiShiVi::CUDA::checkCudaError(cudaMemcpy(h_kdeResult, d_kdeResult, maxNpts * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost), "cudaMemcpy to h_dataSizes error", _debug);

					cudaDeviceSynchronize();

					cudaFree(d_data);
					cudaFree(d_kdeResult);
					cudaFree(d_dataTimes);

					// ПОМЕНЯТЬ ВЫВОД
					for (unsigned int l = 0; l < maxNpts; ++l)
					{
						if (out.is_open())
						{
							if (l != 0 || i != 0)
							{
								out << ", ";
							}
							out << h_kdeResult[l];
						}
						else
						{
							std::cout << "\nOutput file open error" << std::endl;
							exit(1);
						}
					}

					std::free(h_kdeResult);
					std::free(h_dataTimes);
				}
				out << "\n";

				if (_debug)
				{
					/*for (int m = 0; m < (100 / lastIteration); ++m)
						std::cout << '*';*/
					std::cout << "       " << std::setprecision(3) << (100.0f / (float)_nPts) * (i2dIterator + 1) << "%\n";
				}

			}

			//if (_debug)
			//{
			//	//for (int m = 0; m < 100 - ((100 / lastIteration) * lastIteration); ++m)
			//	//	std::cout << '*';
			//	//std::cout << '\n';
			//	std::cout << "       " << "100%\n";
			//	std::cout << '\n';
			//}
			out.close();
		}

		__global__ void roSolve2d(float*		paramValues,
								int			nPts,
								int			TMax,
								float		h,
								float		initialCondition1,
								float		initialCondition2,
								float		initialCondition3,
								int			nValue,
								float		prePeakFinderSliceK,
								float*		data,
								int*		kdeResult,
								float		inA,
								float		inB,
								float		inC,
								int			mode,
								int			kdeSampling,
								float		kdeSamplesInterval1,
								float		kdeSamplesInterval2,
								float		kdeSmoothH)
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

				if (abs(x[nValue]) > 10000.0f)
				{
					kdeResult[idx] = 0;
					return;
				}
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
			//_outSize - кол-во пиков

			float k1 = kdeSampling * _outSize; // K_1
			float k2 = (kdeSamplesInterval2 - kdeSamplesInterval1) / (k1 - 1); // K
			float delt = 0;
			float prevPrevData2 = 0;
			float prevData2 = 0;
			float data2 = 0;
			float memoryData2 = 0;
			bool strangePeak = false;
			int resultKde = 0;

			// Если _outSize == 0
			if (_outSize == 0)
			{
				kdeResult[idx] = 0;
				return;
			}
			// Если _outSize == 1
			if (_outSize == 1)
			{
				kdeResult[idx] = 1;
				return;
			}
			// Если _outSize == 2
			if (_outSize == 2)
			{
				kdeResult[idx] = 1;
				return;
			}

			for (int w = 0; w < k1 - 1; ++w)
			{
				delt = w * k2 + kdeSamplesInterval1;
				prevPrevData2 = prevData2;
				prevData2 = data2;
				data2 = 0;
				for (int m = 0; m < _outSize; ++m)
				{
					float tempData = (data[idx * nIter + m] - delt) / kdeSmoothH;
					data2 += expf(-((tempData * tempData) / 2));
				}
				// Найти здесь - является ли здесь data2 пиком или нет. Если да - инкремируем resultKde
				if (w < 2)
					continue;

				if (strangePeak)
				{
					if (prevData2 == data2)
						continue;
					else if (prevData2 < data2)
					{
						strangePeak = false;
						continue;
					}
					else if (prevData2 > data2)
					{
						strangePeak = false;
						++resultKde;
						continue;
					}
				}
				else if (prevData2 > prevPrevData2 && prevData2 > data2)
				{
					++resultKde;
					continue;
				}
				else if (prevData2 > prevPrevData2 && prevData2 == data2)
				{
					strangePeak = true;
					memoryData2 = prevData2;
					continue;
				}
			}
			if (prevData2 < data2)
			{
				++resultKde;
			}
			kdeResult[idx] = resultKde;
			return;
		}
	}
}