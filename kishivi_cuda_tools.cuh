#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stddef.h>
#include <iostream>

namespace KiShiVi
{
	namespace CUDA
	{
		__host__ void checkCudaError(cudaError_t error, std::string msg, bool debug)
		{
			if (error != cudaSuccess)
			{
				if (debug)
				{
					std::cout << msg;
					std::getchar();
				}
				exit(1);
			}
		}

		__host__ void checkCpuError(void* error, std::string msg, bool debug)
		{
			if (error == NULL)
			{
				if (debug)
				{
					std::cout << msg;
					std::getchar();
				}
				exit(1);
			}
		}

		// Внимание! Функция выделяет память! Не допускать утечку! Р-р-р
		__host__ float* linspace(float start, float finish, int N)
		{
			if (N == 1)
			{
				float* arr = (float*)malloc(sizeof(float));
				arr[0] = start - 1;
				return arr;
			}

			float* arr = (float*)malloc(N * sizeof(float));

			checkCpuError(arr, "linspace", 1);

			float step = (finish - start) / (N - 1);

			for (unsigned int i = 0; i < N; ++i)
				arr[i] = start + (i * step);

			return arr;
		}

		// Внимание! Функция выделяет память! Не допускать утечку! Р-р-р
		__host__ float* slice(float* arr, int start, int finish)
		{
			if (finish == start)
			{
				float* newArr = (float*)malloc(sizeof(float));

				checkCpuError(newArr, "slice error", 1);

				newArr[0] = arr[start - 1];

				return newArr;
			}

			float* newArr = (float*)malloc((finish - start) * sizeof(float));

			checkCpuError(newArr, "slice error", 1);

			int k = 0;
			for (unsigned int i = start; i < finish; ++i)
			{
				newArr[k] = arr[i];
				++k;
			}
			return newArr;
		}
	}
}