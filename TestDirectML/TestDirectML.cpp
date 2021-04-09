// TestDirectML.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "DMLInferer.h"

int main()
{
	try
	{
		DMLInferer inferer;
		inferer.InitD3D12();
		inferer.InitDML();
		inferer.CreateConvolutionOp();
		inferer.InitializeOp();
		inferer.ExecuteOp();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

}
