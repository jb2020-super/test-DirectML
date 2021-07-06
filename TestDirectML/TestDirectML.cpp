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
		//inferer.CreateAddOp(DML_TENSOR_DATA_TYPE_FLOAT16);
		//inferer.CreateOperator(DML_OPERATOR_ELEMENT_WISE_IDENTITY);
		//inferer.CreateOperator(DML_OPERATOR_CONVOLUTION);
		inferer.CreateConvolutionOp(DML_TENSOR_DATA_TYPE_FLOAT16, DML_OPERATOR_ACTIVATION_LEAKY_RELU);
		//inferer.CreateTransposedConvolutionOp(DML_TENSOR_DATA_TYPE_FLOAT16);
		//inferer.CreateTransposedConvolutionOp(DML_TENSOR_DATA_TYPE_FLOAT32/*DML_TENSOR_DATA_TYPE_INT16*/);
		inferer.InitializeOp();
		inferer.ExecuteOp();
		inferer.PrintOutput();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	return 0;
}
