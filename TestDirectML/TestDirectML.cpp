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
		//inferer.CreateOperator(DML_OPERATOR_ELEMENT_WISE_IDENTITY);
		//inferer.CreateOperator(DML_OPERATOR_CONVOLUTION);
		inferer.CreateTransposedConvolutionOp();
		inferer.InitializeOp();
		inferer.ExecuteOp();
		inferer.PrintOutput();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

}
