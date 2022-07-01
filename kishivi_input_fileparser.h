#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>

namespace KiShiVi
{
	namespace FileLib
	{
		//!< ����� ������������ ��� ���������� ������ ��������� �� �������� �����
		//! filePath		- ���� �� �������� ����� (��������: "conf.txt")
		//! parameterName	- �������� ��������� (��������: "PARAM_A")
		//! exitProg		- ���� true - ��������� ��������� � ���������� ��� ������
		//! printResult		- ���� true - �������� ��������� ����������
		//! 
		//! ������� ���������� ������ � �����, ������� ���������� � ������� '\n' ��� '#'
		//! � ������ �� ���������� ��������� ��� ������ �������� ����� ������������ ""
		std::string parseValueFromFile(std::string filePath, std::string parameterName, bool exitProg, bool printResult)
		{
			std::string inBuffer;
			std::ifstream in;
			in.open(filePath);
			if (!in.is_open())
			{
				if (exitProg)
				{
					std::cout << "Input file open error!";
					exit(1);
				}
				return "";
			}

			while (!in.eof())
			{
				std::getline(in, inBuffer);
				if (inBuffer == "\n")
					continue;
				if (inBuffer.size() > 0 && inBuffer[0] == '#')
					continue;
				if (inBuffer.substr(0, inBuffer.find(":")) == parameterName)
				{
					if (printResult)
					{
						std::cout << std::setw(25) << parameterName << ": " << std::setw(7) << inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str() << "\n";
					}
					return inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str();
				}
			}
			if (exitProg)
			{
				std::cout << "Input file error! Not found " << parameterName << " parameter!";
				std::getchar();
				exit(1);
			}
			return "";
		}


	}
}