#ifndef __LW_RNN__
#define __LW_RNN__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>




#define ON_MCU

#ifdef ON_MCU
	#include "stm32h7xx_hal.h"
	#include "lw_init.h"
	#include "lw_usart.h"
#endif


#define FLASH_DATA_BASE_ADDR	0x08020000U 
#define FILE_NAME              "1_weight_bias_data.bin" 



typedef enum
{
	CPU = 1,
	MCU = 2,
	SD = 3
}lw_platform_t;



typedef enum
{
	weight_ih = 0,
	weight_hh = 1,
	bias_ih = 2,
	bias_hh = 3,
	weight_fc = 4,
	bias_fc = 5
}lw_type_t;



typedef enum
{
	xw = 0, 
	hw = 1, 
	fc = 2
}lw_multiply_t;


typedef enum
{
	LAYER_NULL = -1, 
	layer_0 = 0
}lw_layer_t;


typedef struct
{
	float head_len; 
	float input_size;
	float output_size;  
	float hidden_size;  
	float num_layers;  
	float seq_len;  
	float classfication;  
	float para_size; 
}lw_head_t;


float* lw_model(void);

lw_head_t* lw_read_head(void);
float* lw_read_sample(lw_head_t* head);

float* lw_ht_computing(lw_head_t* head, float* test_sample);
float* lw_read_weight_bias(lw_head_t* head, lw_type_t type, lw_layer_t layer);
float* lw_weight_multiply(lw_head_t* head, float* weight, float* x, lw_multiply_t multiply, lw_layer_t layer);
float* lw_read_data(uint32_t len, uint32_t seek);
float* lw_tanh_active(lw_head_t* head, float* bias_out, lw_layer_t layer);
float* lw_layer_bais_add(lw_head_t* head, float* ih_weight_out, float* ih_bias, float* hh_weight_out, float* hh_bias);
float* lw_fc_bais_add(lw_head_t* head, float* fc_weight_out, float* fc_bias);
float* lw_fc_computing(lw_head_t* head, float* ht);

#endif

