
�
Anonymous/my_alinpu_op_0_0HgEnginecommncomm*
InT
2*
OutT
2*

:	�*
output_data_formatNHWC*
data_formatNHWC*?

output_nodes%
#!concat_1/old_output_Quant_PipeEnd*

model_name	default*�"
	all_nodes�"
�"GreaterTileTile/multiplesconcatconcat/axisconcat_1/old_output!concat_1/old_output_Quant_PipeEnd
concat_1/y
dqm_bn/adddqm_bn/add_Quant_PipeEnddqm_bn/mul/_0__cf__0dqm_bn/mul_2(dqm_bn/mul_2_Dequant_IncompletedPipeHeaddqm_bn/sub/_1__cf__1fc1_dqm/MatMulfc1_dqm/MatMul/b,fc1_dqm/MatMul_Dequant_KeepTypeConformance_0fc1_dqm/Neg_1_Mulfc1_dqm/Neg_1_neg_1fc1_dqm/Neg_Mul+fc1_dqm/Neg_Mul_Dequant_IncompletedPipeHeadfc1_dqm/Neg_neg_1fc1_dqm/Relufc1_dqm/Relu_1*fc1_dqm/Relu_1_Dequant_IncompletedPipeHeadfc1_dqm/Relu_1_Quant_PipeEnd:fc1_dqm/Relu_1_Quant_PipeEnd_Dequant_KeepTypeConformance_0fc1_dqm/add
input_nodes0
.Tile_Quant_SegInput0concat_Quant_SegInput1*
qtype

uint16*
op_name
engine_0_0
 
commPlaceholder*
dtype0
�
concat_1	HgDequantAnonymous/my_alinpu_op_0_0*
in_bias
"   �*	
Tin0*
is_excluded(*
is_per_channel( *
data_formatNHWC*
in_scale
"xb�5*

Tout0
!
ncommPlaceholder*
dtype0