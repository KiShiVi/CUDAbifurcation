#include <string>
#include <ctime>
#include <iostream>

#include "../../kishivi_input_fileparser.h"
#include "../../bifurcation2d.cuh"

#define INPUT_FILENAME "conf.txt"

int main()
{
    int         tMax = atoi(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "T_MAX", 1, 1).c_str());                  //!< Время моделирования
    int         nPts = atoi(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "N_PTS", 1, 1).c_str());                  //!< Кол-во точек
    float       h = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "H", 1, 1).c_str());                      //!< Шаг интегрирования

    float       initialCondition1 = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "INITIAL_CONDITIONS_1", 1, 1).c_str());   //!< Начальные условия x
    float       initialCondition2 = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "INITIAL_CONDITIONS_2", 1, 1).c_str());   //!< Начальные условия y
    float       initialCondition3 = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "INITIAL_CONDITIONS_3", 1, 1).c_str());   //!< Начальные условия z

    float       paramValues1 = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "PARAM_VALUES_1", 1, 1).c_str());         //!< Начало диапазона расчета
    float       paramValues2 = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "PARAM_VALUES_2", 1, 1).c_str());         //!< Конец диапазона расчета

    float       paramValues3 = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "PARAM_VALUES_3", 1, 1).c_str());         //!< Конец диапазона расчета
    float       paramValues4 = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "PARAM_VALUES_4", 1, 1).c_str());         //!< Конец диапазона расчета

    int         nValue = atoi(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "N_VALUE", 1, 1).c_str());                //!< Какую координату (0/1/2 = x/y/z) берем в расчет
    float       prePeakFinderSliceK = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "PRE_PEAKFINDER_SLICE_K", 1, 1).c_str()); //!< Какой процент точек отрезаем (отсекам переходный процесс)

    float       paramA = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "PARAM_A", 1, 1).c_str());                //!< Параметр A
    float       paramB = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "PARAM_B", 1, 1).c_str());                //!< Параметр B
    float       paramC = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "PARAM_C", 1, 1).c_str());                //!< Параметр C

    int         mode  = atoi(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "MODE_1", 1, 1).c_str());                   //!< По какому параметру обходим (см. enum Mode)
    int         mode2 = atoi(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "MODE_2", 1, 1).c_str());                   //!< По какому параметру обходим во второй плоскости (см. enum Mode)

    int         kdeSampling = atoi(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "KDE_SAMPLING", 1, 1).c_str());           // Какой-то sampling из kde

    float       kdeSamplesInterval1 = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "KDE_SAMPLES_INTERVAL_1", 1, 1).c_str());          // Какой-то SAMPLES_INTERVAL
    float       kdeSamplesInterval2 = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "KDE_SAMPLES_INTERVAL_2", 1, 1).c_str());          // Какой-то SAMPLES_INTERVAL

    float       kdeSmoothH = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "KDE_SMOOT_H", 1, 1).c_str());          // Какой-то SAMPLES_INTERVAL

    float       memoryLimit = atof(KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "MEMORY_LIMIT", 1, 1).c_str());           //!< Какой-то kdeSmoothH

    std::string outPath = KiShiVi::FileLib::parseValueFromFile(INPUT_FILENAME, "OUTPATH", 1, 1);                              //!< Путь к файлу с результатом 

    size_t startTime = clock();

    KiShiVi::CUDA::bifurcation2d(tMax,
        nPts,
        h,
        initialCondition1,
        initialCondition2,
        initialCondition3,
        paramValues1,
        paramValues2,
        paramValues3,
        paramValues4,
        nValue,
        prePeakFinderSliceK,
        paramA,
        paramB,
        paramC,
        mode,
        mode2,
        memoryLimit,
        kdeSampling,
        kdeSamplesInterval1,
        kdeSamplesInterval2,
        kdeSmoothH,
        outPath,
        1);

    std::cout << clock() - startTime << " ms";
    std::getchar();
    return 0;
}