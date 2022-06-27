#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>

/**************   MAIN SETTINGS   ***************/

#define T_MAX                   1000
#define H                       0.01
#define N_PTS                   1000
#define INITIAL_CONDITIONS_1    0.1
#define INITIAL_CONDITIONS_2    0.1
#define INITIAL_CONDITIONS_3    0.1

#define PARAM_VALUES_1          0.05
#define PARAM_VALUES_2          0.35

#define N_VALUE                 1

/************************************************/
/**************   FUNC SETTINGS   ***************/

#define PARAM_A 0.2
#define PARAM_B 0.2
#define PARAM_C 5.7

/************************************************/


void roSolve(double s, int TMax, double* initialConditions, double h, double**& Y);
double* linspace(double start, double finish, int N);
void slice(double*& arr, int size, int start, int finish);
void peakFinder(double*& arr, int arrSize, double*& out, int& outSize);

int main()
{
    int                     TMax                = T_MAX;
    double                  h                   = H;
    int                     NPts                = N_PTS;

    double*                 initialConditions   = (double*)malloc(3 * sizeof(double));
    initialConditions[0]                        = INITIAL_CONDITIONS_1;
    initialConditions[1]                        = INITIAL_CONDITIONS_2;
    initialConditions[2]                        = INITIAL_CONDITIONS_3;

    double*                 paramValues         = linspace(PARAM_VALUES_1, PARAM_VALUES_2, NPts);
    int                     nValue              = N_VALUE;

    int                     YSize               = TMax / h;

    std::ofstream out;          // поток для записи
    out.open("C:\\Users\\kshir\\Desktop\\mat.csv"); // окрываем файл для записи

    for (int i = 0; i < NPts; ++i)
    {
        double s = paramValues[i];

        double** Y = nullptr;

        roSolve(s, TMax, initialConditions, h, Y);

        slice(Y[0], YSize, 0.7 * YSize, YSize);
        slice(Y[1], YSize, 0.7 * YSize, YSize);
        slice(Y[2], YSize, 0.7 * YSize, YSize);

        double* peakMagr;
        int peakMagrSize;
        peakFinder(Y[N_VALUE], YSize - 0.7 * YSize, peakMagr, peakMagrSize);

        free(Y[0]);
        free(Y[1]);
        free(Y[2]);
        free(Y);

        slice(peakMagr, peakMagrSize, 12, peakMagrSize);


        if (out.is_open())
        {
            for (int i = 0; i < peakMagrSize - 12; ++i)
                out << s << ", " << peakMagr[i] << std::endl;
        }

        free(peakMagr);

    }
    out.close();
    return 0;
}

void roSolve(double s, int TMax, double* initialConditions, double h, double**& Y)
{
    int nIter   = TMax / h;
    double localH1, localH2;
    double* x = (double*)malloc(3 * sizeof(double));
    for (int i = 0; i < 3; ++i)
        x[i] = initialConditions[i];

    localH1 = h * 0.5;
    localH2 = h * (1 - 0.5);

    double** local_Y = (double**)malloc(3 * sizeof(double*));

    for (int i = 0; i < 3; ++i)
    {
        local_Y[i] = (double*)malloc(nIter * sizeof(double));
        for (int j = 0; j < nIter; ++j)
            local_Y[i][j] = 0;
    }

    double a = s;
    double b = PARAM_B;
    double c = PARAM_C;

    for (int i = 0; i < nIter; ++i)
    {
        for (int j = 0; j < 3; ++j)
            local_Y[j][i] = x[j];

        x[0] = x[0] + localH1 * ( -x[1] - x[2] );
        x[1] = ( x[1] + localH1 * ( x[0] ) ) / ( 1 - a * localH1 );
        x[2] = ( x[2] + localH1 * b ) / ( 1 - localH1 * ( x[0] - c ) );

        x[2] = x[2] + localH2 * ( b + x[2] * ( x[0] - c ) );
        x[1] = x[1] + localH2 * ( x[0] + a * x[1] );
        x[0] = x[0] + localH2 * ( -x[1] - x[2] );
    }

    Y = local_Y;

    free(x);

    return;
}

void slice(double*& arr, int size, int start, int finish)
{
    double* newArr = (double*)malloc((finish - start + 1) * sizeof(double));
    for (int i = 0; i < (finish - start + 1); ++i)
        newArr[i] = arr[i + start];
    free(arr);
    arr = newArr;
    return;
}

void peakFinder(double*& arr, int arrSize, double*& out, int& outSize)
{
    out = (double*)malloc(arrSize * sizeof(double));
    int _outSize = 0;
    for (int i = 1; i < arrSize - 1; ++i)
    {
        if (arr[i] > arr[i - 1] && arr[i] > arr[i + 1])
        {
            out[_outSize] = arr[i];
            ++_outSize;
        }
    }
    realloc(out, _outSize * sizeof(double));
    outSize = _outSize;
}

double* linspace(double start, double finish, int N)
{
    if (N < 2)
    {
        throw "linspace() error - { N < 2 }";
        exit(1);
    }

    double* arr = (double*)malloc(N * sizeof(double));
    double step = (finish - start) / (N - 1);

    for (int i = 0; i < N; ++i)
        arr[i] = start + (i * step);

    return arr;
}
