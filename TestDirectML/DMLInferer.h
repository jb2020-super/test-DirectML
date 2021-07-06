#pragma once
#include "common.h"

class DMLTensor 
{
public:
	DMLTensor();
	~DMLTensor();
	void Create(DML_TENSOR_DATA_TYPE data_type, const UINT* dims, UINT dim_cnt, bool is_interleaved);
	
	DML_TENSOR_DESC& GetDesc();
	UINT64 GetTensorSizeInBytes();
	UINT GetElementCount();
	UINT DimCount();
	UINT Width();
	UINT Height();

private:
	
	UINT m_dims[5];
	UINT m_input_strides[4]{};
	DML_BUFFER_TENSOR_DESC m_buffer_desc;
	DML_TENSOR_DESC m_desc;
};

class DMLInferer
{
public:
	DMLInferer();
	~DMLInferer();
	void CreateAddOp(DML_TENSOR_DATA_TYPE dtype);
	void CreateOperator(DML_OPERATOR_TYPE op_type);
	void CreateConvolutionOp(DML_TENSOR_DATA_TYPE dtype, DML_OPERATOR_TYPE op_type);
	void CreateIdentityOp();
	void CreateTransposedConvolutionOp(DML_TENSOR_DATA_TYPE data_type);
	void ExecuteOp();
	void InitD3D12(bool use_warp=false);
	void InitDML();
	void InitializeOp();
	void PrintOutput();
private:
	void _create_resource(void* data, UINT64 buffer_size, winrt::com_ptr<ID3D12Resource>& rc, winrt::com_ptr<ID3D12Resource>& upload_rc);
	void _close_execute();
	template<class DataType>
	void _print_tensor(const char* tensor_name, DMLTensor& tensor, DataType* data);
	void _upload_convolution_data();
	void _upload_identity_data();
	void _upload_add_data();
	UINT _cal_transposed_conv_2d_out_size(UINT input, UINT stride, UINT pad, UINT dilation, UINT kernel, UINT out_pad);

	DML_OPERATOR_TYPE m_op_type{};
	winrt::com_ptr<ID3D12Device> m_device;
	winrt::com_ptr<ID3D12CommandQueue> m_cmd_queue;
	winrt::com_ptr<ID3D12CommandAllocator> m_cmd_allocator;
	winrt::com_ptr<ID3D12GraphicsCommandList> m_cmd_list;
	winrt::com_ptr<IDMLDevice> m_dml_device;
	winrt::com_ptr<IDMLOperator> m_dml_op;
	winrt::com_ptr<IDMLCompiledOperator> m_pso;
	winrt::com_ptr<IDMLOperatorInitializer> m_op_initer;
	winrt::com_ptr<ID3D12DescriptorHeap> m_dsp_heap;
	winrt::com_ptr<IDMLBindingTable> m_binding_tlb;
	winrt::com_ptr<IDMLCommandRecorder> m_recoder;
	winrt::com_ptr<ID3D12Resource> m_input_rc;
	winrt::com_ptr<ID3D12Resource> m_filter_rc;
	winrt::com_ptr<ID3D12Resource> m_output_rc;
	winrt::com_ptr<ID3D12Resource> m_tmp_buffer;
	winrt::com_ptr<ID3D12Resource> m_presis_buffer;
	winrt::com_ptr<ID3D12Resource> m_upload_rc;
	winrt::com_ptr<ID3D12Resource> m_upload_rc2;
	winrt::com_ptr<IDXGraphicsAnalysis> m_dxgraphics_analysis;
	DMLTensor m_input_tensor;
	DMLTensor m_filter_tensor;
	DMLTensor m_output_tensor;
	DML_TENSOR_DATA_TYPE m_data_type{};
	bool m_is_backward{ false };
	bool m_enable_print{ false };
	bool m_is_nvidia{ false };
};

template<class DataType>
inline void DMLInferer::_print_tensor(const char* tensor_name, DMLTensor& tensor, DataType* data)
{
	if (m_enable_print) {
		int height = tensor.Height();
		int width = tensor.Width();
		std::cout << tensor_name << " tensor dimension: " << height << "x" << width << std::endl;
		for (int i = 0, idx = 0; i < tensor.Height(); ++i) {
			for (int j = 0; j < tensor.Width(); ++j) {
				std::cout << data[idx++] << "\t";
			}
			std::cout << std::endl;
		}
	}
}
