#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <iomanip>

/**************   MAIN SETTINGS   ***************/

#define T_MAX                           1000
#define H                               0.01
#define N_PTS                           10000
#define INITIAL_CONDITIONS_1            0.1
#define INITIAL_CONDITIONS_2            0.1
#define INITIAL_CONDITIONS_3            0.1

#define PARAM_VALUES_1                  0.05
#define PARAM_VALUES_2                  0.35

#define N_VALUE                         1

#define PRE_PEAKFINDER_SLICE_K          0.3      
#define PEAKFINDER_SLICE_POINT_COUNT    12
/************************************************/
/**************   FUNC SETTINGS   ***************/

#define PARAM_A 0.2
#define PARAM_B 0.2
#define PARAM_C 5.7

/************************************************/

enum Mode 
{
    SYMMETRY_MODE,
    PARAM_A_MODE,
    PARAM_B_MODE,
    PARAM_C_MODE
};

void gpu( int _tMax,
          int _nPts,
          float _h,
          float _initialCondition1,
          float _initialCondition2,
          float _initialCondition3,
          float _paramValues1,
          float _paramValues2,
          int _nValue,
          float _prePeakFinderSliceK,
          float _paramA,
          float _paramB,
          float _paramC,
          int _mode,
          std::string _outPath );
float* linspace(float start, float finish, int N);
float* slice(float* arr, int start, int finish);
__global__ void roSolve(float* paramValues,
                        int        nPts,
                        int        TMax,
                        float     h,
                        float* initialConditions,
                        int        nValue,
                        float     prePeakFinderSliceK,
                        int        peakFinderSlicePointCount,
                        /*float* rawData,*/
                        float* data,
                        int* dataSizes,
                        float* dataTimes,
                        float      inA,
                        float      inB,
                        float      inC,
                        int        mode );
int main()
{
    int tMax;
    int nPts;
    float h;

    float initialCondition1;
    float initialCondition2;
    float initialCondition3;

    float paramValues1;
    float paramValues2;

    int nValue;
    float prePeakFinderSliceK;

    float paramA;
    float paramB;
    float paramC;

    int mode;

    std::string outPath;

    std::string inBuffer;

    std::ifstream in;
    std::cout << "Reading conf.txt...\n";
    in.open("conf.txt");
    if (!in.is_open())
    {
        std::cout << "Input file opennig error\n";
        exit(1);
    }
    while (!in.eof())
    {
        std::getline(in, inBuffer);
        if (inBuffer == "\n")
            continue;
        if (inBuffer.size() > 0 && inBuffer[0] == '#')
            continue;

        if (inBuffer.substr(0, inBuffer.find(":")) == "T_MAX")
            tMax = std::atoi(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());
        else if (inBuffer.substr(0, inBuffer.find(":")) == "N_PTS")
            nPts = std::atoi(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());
        else if (inBuffer.substr(0, inBuffer.find(":")) == "H")
            h = std::atof(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());

        else if (inBuffer.substr(0, inBuffer.find(":")) == "INITIAL_CONDITIONS_1")
            initialCondition1 = std::atof(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());
        else if (inBuffer.substr(0, inBuffer.find(":")) == "INITIAL_CONDITIONS_2")
            initialCondition2 = std::atof(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());
        else if (inBuffer.substr(0, inBuffer.find(":")) == "INITIAL_CONDITIONS_3")
            initialCondition3 = std::atof(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());

        else if (inBuffer.substr(0, inBuffer.find(":")) == "PARAM_VALUES_1")
            paramValues1 = std::atof(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());
        else if (inBuffer.substr(0, inBuffer.find(":")) == "PARAM_VALUES_2")
            paramValues2 = std::atof(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());

        else if (inBuffer.substr(0, inBuffer.find(":")) == "N_VALUE")
            nValue = std::atoi(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());

        else if (inBuffer.substr(0, inBuffer.find(":")) == "PRE_PEAKFINDER_SLICE_K")
            prePeakFinderSliceK = std::atof(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());

        else if (inBuffer.substr(0, inBuffer.find(":")) == "PARAM_A")
            paramA = std::atof(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());
        else if (inBuffer.substr(0, inBuffer.find(":")) == "PARAM_B")
            paramB = std::atof(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());
        else if (inBuffer.substr(0, inBuffer.find(":")) == "PARAM_C")
            paramC = std::atof(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());

        else if (inBuffer.substr(0, inBuffer.find(":")) == "MODE")
            mode = std::atoi(inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str());

        else if (inBuffer.substr(0, inBuffer.find(":")) == "OUTPATH")
            outPath = inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str();

        else
            continue;
    }

    in.close();

    std::cout << "Successful reading!\n\n";

    std::cout << std::setw(32) << std::setfill('#');

    std::cout << "\n" << std::setfill(' ');

    std::cout << std::setw(25) << "T_MAX: "                     << std::setw(7) << tMax                    << "\n";
    std::cout << std::setw(25) << "N_PTS: "                     << std::setw(7) << nPts                    << "\n";
    std::cout << std::setw(25) << "H: "                         << std::setw(7) << h                       << "\n";
    std::cout << std::setw(25) << "INITIAL_CONDITIONS_2: "      << std::setw(7) << initialCondition1       << "\n";
    std::cout << std::setw(25) << "INITIAL_CONDITIONS_2: "      << std::setw(7) << initialCondition2       << "\n";
    std::cout << std::setw(25) << "INITIAL_CONDITIONS_2: "      << std::setw(7) << initialCondition3       << "\n";
    std::cout << std::setw(25) << "PARAM_VALUES_1: "            << std::setw(7) << paramValues1            << "\n";
    std::cout << std::setw(25) << "PARAM_VALUES_2: "            << std::setw(7) << paramValues2            << "\n";
    std::cout << std::setw(25) << "N_VALUE: "                   << std::setw(7) << nValue                  << "\n";
    std::cout << std::setw(25) << "PRE_PEAKFINDER_SLICE_K: "    << std::setw(7) << prePeakFinderSliceK     << "\n";
    std::cout << std::setw(25) << "PARAM_A: "                   << std::setw(7) << paramA                  << "\n";
    std::cout << std::setw(25) << "PARAM_A: "                   << std::setw(7) << paramB                  << "\n";
    std::cout << std::setw(25) << "PARAM_A: "                   << std::setw(7) << paramC                  << "\n";
    std::cout << std::setw(25) << "MODE: "                      << std::setw(7) << mode                    << "\n";
    std::cout << std::setw(25) << "OUTPATH: "                   << outPath                                 << "\n";

    std::cout << "\n";

    std::cout << std::setw(32) << std::setfill('#');

    std::cout << "\n\n" << std::setfill(' ');

    unsigned int time = clock();

    std::cout << "Calculating...\n";

    gpu( tMax,
         nPts,
         h,
         initialCondition1,
         initialCondition2,
         initialCondition3,
         paramValues1,
         paramValues2,
         nValue,
         prePeakFinderSliceK,
         paramA,
         paramB,
         paramC,
         mode,
         outPath );

    std::cout << "Successful!\n";
    std::cout << clock() - time << " ms\n";
    std::getchar();
    return 0;
}

void gpu( int _tMax,
          int _nPts,
          float _h,
          float _initialCondition1,
          float _initialCondition2,
          float _initialCondition3,
          float _paramValues1,
          float _paramValues2,
          int _nValue,
          float _prePeakFinderSliceK,
          float _paramA,
          float _paramB,
          float _paramC,
          int _mode,
          std::string _outPath)
{
    int         TMax                = _tMax;
    float       h                   = _h;
    int         globalNPts          = _nPts;

    float* initialConditions       = (float*)malloc(3 * sizeof(float));
    initialConditions[0]            = _initialCondition1;
    initialConditions[1]            = _initialCondition2;
    initialConditions[2]            = _initialCondition3;

    float* globalParamValues = linspace(_paramValues1, _paramValues2, globalNPts);

    int                         nValue  = _nValue;
    unsigned long long          YSize   = TMax / h;

    bool isWorking = true;

    int k = 0;

    int NPts            = globalNPts;
    int remainderNPts   = NPts;


    std::ofstream out;          // поток для записи
    out.open(_outPath); // окрываем файл для записи

    while (isWorking)
    { 
        // 10000000 надо будет как-то вычлинять исходя из характеристик устройства.
        float* paramValues = nullptr;
        if (remainderNPts > (10000000 / TMax))
        {
            NPts = 10000000 / TMax;
            remainderNPts -= NPts;
            paramValues = slice(globalParamValues, k * (10000000 / TMax), (k + 1) * (10000000 / TMax));
        }
        else
        {
            isWorking = false;
            NPts = globalNPts - (k * (10000000 / TMax));
            paramValues = slice(globalParamValues, globalNPts - NPts, globalNPts /* + 1*/);
        }
        
        float * d_paramValues;
        float * d_initialConditions;
        //float * d_rawData;
        float * d_data;
        int * d_dataSizes;
        float* d_dataTimes;

        if (cudaSuccess != cudaMalloc((void**)&d_paramValues, NPts * sizeof(float)))
        {
            std::cout << "1 Error\n";
            exit(1);
        }
        if (cudaSuccess != cudaMalloc((void**)&d_initialConditions, 3 * sizeof(float)))
        {
            std::cout << "2 Error\n";
            exit(1);
        }
        if (cudaSuccess != cudaMalloc((void**)&d_data, YSize * NPts * sizeof(float)))
        {
            std::cout << "3 Error\n";
            exit(1);
        }
        if (cudaSuccess != cudaMalloc((void**)&d_dataSizes, NPts * sizeof(int)))
        {
            std::cout << "4 Error\n";
            exit(1);
        }
        if (cudaSuccess != cudaMalloc((void**)&d_dataTimes, NPts * sizeof(float)))
        {
            std::cout << "5 Error\n";
            exit(1);
        }

        if (cudaSuccess != cudaMemcpy(d_paramValues,       paramValues,        NPts * sizeof(float),  cudaMemcpyKind::cudaMemcpyHostToDevice))
        {
            std::cout << "6 Error\n";
            exit(1);
        }
        if (cudaSuccess != cudaMemcpy(d_initialConditions, initialConditions,  3 * sizeof(float),     cudaMemcpyKind::cudaMemcpyHostToDevice))
        {
            std::cout << "7 Error\n";
            exit(1);
        }



        int blockSize;
        int minGridSize;
        int gridSize;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, roSolve, 0, NPts);
        gridSize = (NPts + blockSize - 1) / blockSize;
        roSolve<<<gridSize, blockSize>>> ( d_paramValues,
                                           NPts,
                                           TMax,
                                           h,
                                           d_initialConditions,
                                           _nValue,
                                           _prePeakFinderSliceK,
                                           0,
                                           /*d_rawData,*/
                                           d_data,
                                           d_dataSizes,
                                           d_dataTimes,
                                           _paramA,
                                           _paramB,
                                           _paramC,
                                           _mode );

        cudaDeviceSynchronize();

        float* data        = (float*)malloc(YSize * NPts * sizeof(float));
        if (data == NULL)
        {
            std::cout << "8 Error\n";
            exit(1);
        }
        int* dataSizes     = (int*)malloc(NPts * sizeof(int));
        if (dataSizes == NULL)
        {
            std::cout << "9 Error\n";
            exit(1);
        }
        float* dataTimes   = (float*)malloc(NPts * sizeof(float));
        if (dataTimes == NULL)
        {
            std::cout << "10 Error\n";
            exit(1);
        }
    
        if (cudaSuccess != cudaMemcpy(dataSizes,   d_dataSizes,    NPts * sizeof(int),             cudaMemcpyKind::cudaMemcpyDeviceToHost))
        {
            std::cout << "11 Error\n";
            exit(1);
        }
        if (cudaSuccess != cudaMemcpy(dataTimes,   d_dataTimes,    NPts * sizeof(float),          cudaMemcpyKind::cudaMemcpyDeviceToHost))
        {
            std::cout << "12 Error\n";
            exit(1);
        }
        if (cudaSuccess != cudaMemcpy(data, d_data, YSize * NPts * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost))
        {
            std::cout << "13 Error\n";
            exit(1);
        }

        cudaFree(d_paramValues);
        cudaFree(d_initialConditions);
        //cudaFree(d_rawData);
        cudaFree(d_data);
        cudaFree(d_dataSizes);
        cudaFree(d_dataTimes);

        //for (int i = 0; i < 100; ++i)
        //    std::cout << "#" << i << ": " << dataTimes[i] << std::endl;

        for (unsigned int i = 0; i < NPts; ++i)
        {
            //std::cout << "#" << i << std::endl;
            for (unsigned int j = 0; j < dataSizes[i]; ++j)
            {
                if (out.is_open())
                {
                    //std::cout << dataTimes[i] << ", " << data[i * YSize + j] << std::endl;
                    out << dataTimes[i] << ", " << data[i * YSize + j] << '\n';
                }
                else
                {
                    std::cout << "FILE OPENNING ERROR" << std::endl;
                }
            }
        }
        free(data);
        free(dataSizes);
        free(dataTimes);
        delete[] paramValues;
        ++k;

        //копирование данных

        //Создание выходных массивов. Выделение памяти в GPU
        //Расчет сетки блоков и потоков
        //call gpuMethod()
    }
    out.close();
}

// paramValues                  - Основной параметр прогона = S                     -> in (malloc)
// nPts                         - Кол-во прогонов                                   -> in
// TMax                         - Время моделирования                               -> in
// h                            - Шаг интегрирования                                -> in
// initialConditions            - Начальные условия                                 -> in (malloc)
// nValue                       - Просматриваемый параметр                          -> in
// prePeakFinderSliceK          - Коэф-нт отрезки ненужных точек перед peakFinder   -> in
// peakFinderSlicePointCount    - Кол-во точек отрезки точек после peakFinder       -> in
// /*rawData                      - Наайденные точки ДО peakFinder'a.                 -> ~out~ (malloc) {size = nPts * nIter}*/
// data                         - Результат. ВСЕ найденные точки ПОСЛЕ peakFinder   -> out (malloc) {size = nPts * nIter}
// dataSizes                    - Размеры блоков в массиве data                     -> out (malloc)
// dataTimes                    - Время каждого из блоков в массиве data            -> out (malloc)

__global__ void roSolve( float*     paramValues,
                         int        nPts,
                         int        TMax,
                         float      h,
                         float*     initialConditions,
                         int        nValue,
                         float      prePeakFinderSliceK,
                         int        peakFinderSlicePointCount,
                         /*float*    rawData,*/
                         float*     data,
                         int*       dataSizes,
                         float*     dataTimes,
                         float      inA,
                         float      inB,
                         float      inC,
                         int        mode )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= nPts)
        return;
    float s = paramValues[idx];

    /***********************************************/

    int nIter = TMax / h;
    float localH1, localH2;
    float x[3]{ initialConditions[0], initialConditions[1], initialConditions[2] };

    if ( mode == SYMMETRY_MODE )
    {
        localH1 = h * s;
        localH2 = h * (1 -s);
    }
    else
    {
        localH1 = h * 0.5;
        localH2 = h * (1 - 0.5);
    }

    // Здесь подобные выходки кажутся дерзкими и не приличными по причине "причина"
    // Комент на подумать в будущем
    //for (int i = 0; i < nPts * nIter; ++i)
    //    data[i] = 0;

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

    //float * rawData = new float[nIter * nPts];

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
    for (unsigned int i = 1 + (prePeakFinderSliceK * (TMax / h)); i < (TMax / h) - 1; ++i)
    {
        if (data[idx * nIter + i] > data[idx * nIter + i - 1] && data[idx * nIter + i] > data[idx * nIter + i + 1])
        {
            data[idx * nIter + _outSize] = data[idx * nIter + i];
            ++_outSize;
        }        
    }
    dataSizes[idx] = _outSize;
    dataTimes[idx] = s;
}

float* linspace(float start, float finish, int N)
{
    if (N < 2)
    {
        throw "linspace() error - { N < 2 }";
        exit(1);
    }

    float* arr = (float*)malloc(N * sizeof(float));
    float step = (finish - start) / (N - 1);

    for (unsigned int i = 0; i < N; ++i)
        arr[i] = start + (i * step);

    return arr;
}

float* slice(float* arr, int start, int finish)
{
    float* newArr = (float*)malloc((finish-start) * sizeof(float));
    int k = 0;
    for(unsigned int i = start; i < finish; ++i)
    {
        newArr[k] = arr[i];
        ++k;
    }
    return newArr;
}