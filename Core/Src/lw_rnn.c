#include "lw_rnn.h"


float* lw_model(void)
{
	lw_head_t*  head = lw_read_head(); 
	float* test_sample = lw_read_sample(head); 
	float* ht = lw_ht_computing(head, test_sample);
	float* fc_out = lw_fc_computing(head, ht);

	return fc_out;
}


float* lw_fc_computing(lw_head_t* head, float* ht)
{
	
	float* fc_weight = lw_read_weight_bias(head, weight_fc, LAYER_NULL);
	float* fc_weight_out = lw_weight_multiply(head, fc_weight, ht, fc, LAYER_NULL);

	
	float* fc_bias = lw_read_weight_bias(head, bias_fc, LAYER_NULL);
	float* bias_out = lw_fc_bais_add(head, fc_weight_out, fc_bias); 

	free(head);
	head = NULL;
	return bias_out;
}



float* lw_ht_computing(lw_head_t* head, float* test_sample)
{
	uint32_t ht_len = (uint32_t)head->hidden_size; 
	uint32_t ht_len_bytes = sizeof(float) * ht_len; 
	float* ht = (float*)calloc((uint32_t)(ht_len_bytes * head->num_layers), 1); 

	for (uint32_t t = 0; t < (uint32_t)head->seq_len; t++) 
	{
		
		for (uint32_t i = 0; i < (uint32_t)head->num_layers; i++) 
		{
			
			float* indata = NULL; 
			if (i == 0) 
			{
				indata = (float*)calloc(sizeof(float), 1); 
				memmove(indata, &test_sample[t], sizeof(float));
			}
			else
			{
				indata = (float*)calloc((uint32_t)ht_len_bytes, 1); 
				memmove(indata, ht + (i - 1) * ht_len, ht_len_bytes);
			}
			float* ih_weight = lw_read_weight_bias(head, weight_ih, (lw_layer_t)(i));
			float* ih_weight_out = lw_weight_multiply(head, ih_weight, indata, xw, (lw_layer_t)(i));

			
			float* ht_1 = (float*)calloc((uint32_t)ht_len_bytes, 1); 
			if (t != 0) 
			{
				if (i == 0) 
				{
					memmove(ht_1, ht, ht_len_bytes); 
				}
				else
				{
					memmove(ht_1, ht + i * ht_len, ht_len_bytes);
				}
			}
			float* hh_weight = lw_read_weight_bias(head, weight_hh, (lw_layer_t)(i));
			float* hh_weight_out = lw_weight_multiply(head, hh_weight, ht_1, hw, (lw_layer_t)(i));

			float* ih_bias = lw_read_weight_bias(head, bias_ih, (lw_layer_t)(i));
			float* hh_bias = lw_read_weight_bias(head, bias_hh, (lw_layer_t)(i));
			float* bias_out = lw_layer_bais_add(head, ih_weight_out, ih_bias, hh_weight_out, hh_bias);

			float* tanh_out = lw_tanh_active(head, bias_out, (lw_layer_t)(i)); 
			memmove(ht + (i * ht_len), tanh_out, ht_len_bytes); 
			free(tanh_out);
			tanh_out = NULL;
		}
	}
	free(test_sample);
	test_sample = NULL;

	float* ht_out = (float*)calloc((uint32_t)(ht_len_bytes), 1); 
	memmove(ht_out, ht + (uint32_t)(head->num_layers - 1) * ht_len, ht_len_bytes);

	return ht_out;
}


float* lw_read_weight_bias(lw_head_t* head, lw_type_t type, lw_layer_t layer)
{
	uint32_t len = 0;
	uint32_t seek = 0;
	uint32_t offset = 0;

	uint32_t weight_ih_len = 0;
	uint32_t weight_hh_len = 0;
	uint32_t bais_ih_len = 0;
	uint32_t bais_hh_len = 0;

	
	if ((type == weight_fc) || (type == bias_fc))
	{
		switch (type)
		{
		case weight_fc:
			len = (uint32_t)(sizeof(float) * head->hidden_size * head->classfication);
			seek = (uint32_t)(sizeof(float) * (head->head_len + head->para_size - head->classfication - head->hidden_size * head->classfication)); 
			break;

		case bias_fc:
			len = (uint32_t)(sizeof(float) * head->classfication);
			seek = (uint32_t)(sizeof(float) * (head->head_len + head->para_size - head->classfication)); 
			break;
		}
	}
	else
	{
		
		if (layer == layer_0) 
		{
			switch (type)
			{
			case weight_ih:
				len = (uint32_t)(sizeof(float) * head->hidden_size);
				seek = (uint32_t)(sizeof(float) * head->head_len);
				break;

			case weight_hh:
				len = (uint32_t)(sizeof(float) * head->hidden_size * head->hidden_size);
				seek = (uint32_t)(sizeof(float) * (head->head_len + head->hidden_size));
				break;

			case bias_ih:
				len = (uint32_t)(sizeof(float) * head->hidden_size);
				seek = (uint32_t)(sizeof(float) * (head->head_len + head->hidden_size + head->hidden_size * head->hidden_size));
				break;

			case bias_hh:
				len = (uint32_t)(sizeof(float) * head->hidden_size);
				seek = (uint32_t)(sizeof(float) * (head->head_len + head->hidden_size + head->hidden_size * head->hidden_size + head->hidden_size));
				break;
			}
		}
		else
		{
			
			for (uint16_t i = 0; i < (uint16_t)(layer); i++)
			{
				if (i == 0) 
				{
					weight_ih_len = (uint32_t)(head->hidden_size * head->input_size);
				}
				else
				{
					weight_ih_len = (uint32_t)(head->hidden_size * head->hidden_size);
				}

				weight_hh_len = (uint32_t)(head->hidden_size * head->hidden_size);
				bais_ih_len = (uint32_t)(head->hidden_size);
				bais_hh_len = (uint32_t)(head->hidden_size);

				offset += (uint32_t)(weight_ih_len + weight_hh_len + bais_ih_len + bais_hh_len);
			}
			
			switch (type)
			{
			case weight_ih:
				offset += 0; 
				break;

			case weight_hh:
				offset += (uint32_t)(head->hidden_size * head->hidden_size); 
				break;

			case bias_ih:
				offset += (uint32_t)(head->hidden_size * head->hidden_size); 
				offset += (uint32_t)(head->hidden_size * head->hidden_size); 
				break;

			case bias_hh:
				offset += (uint32_t)(head->hidden_size * head->hidden_size); 
				offset += (uint32_t)(head->hidden_size * head->hidden_size); 
				offset += (uint32_t)(head->hidden_size); 
				break;

			case weight_fc:
				offset += (uint32_t)(head->hidden_size * head->hidden_size); 
				offset += (uint32_t)(head->hidden_size * head->hidden_size); 
				offset += (uint32_t)(head->hidden_size); 
				offset += (uint32_t)(head->hidden_size); 
				break;

			case bias_fc:
				offset += (uint32_t)(head->hidden_size * head->hidden_size); 
				offset += (uint32_t)(head->hidden_size * head->hidden_size); 
				offset += (uint32_t)(head->hidden_size); 
				offset += (uint32_t)(head->hidden_size); 
				offset += (uint32_t)(head->hidden_size * head->classfication); 
				break;
			}

			len = (uint32_t)(sizeof(float) * weight_hh_len);
			seek = (uint32_t)(sizeof(float) * (head->head_len + offset));
		}
	}

	float* weight_bias = lw_read_data(len, seek);
	return weight_bias;
}


float* lw_weight_multiply(lw_head_t* head, float* weight, float* x, lw_multiply_t multiply, lw_layer_t layer)
{
	uint16_t row = (uint16_t)head->hidden_size; 
	uint16_t col = (uint16_t)head->hidden_size; 

	
	if ((multiply == xw) && (layer == layer_0)) 
	{
		col = (uint16_t)head->input_size; 
	}

	
	if (multiply == fc) 
	{
		row = (uint16_t)head->classfication; 
	}

	
	float* weight_multiply_out = (float*)calloc(sizeof(float) * row, 1); 

	if (weight_multiply_out != NULL)
	{
		for (uint32_t i = 0; i < row; i++) 
		{
			float sum = 0;
			for (uint32_t j = 0; j < col; j++) 
			{
				sum += weight[col * i + j] * x[j];
			}
			weight_multiply_out[i] = sum;
		}
	}

	free(weight);
	free(x);
	weight = NULL;
	x = NULL;
	return weight_multiply_out;
}


float* lw_layer_bais_add(lw_head_t* head, float* ih_weight_out, float* ih_bias, float* hh_weight_out, float* hh_bias)
{
	float* bias_out = (float*)calloc((uint32_t)(sizeof(float) * head->hidden_size), 1); 

	for (uint32_t i = 0; i < (uint32_t)head->hidden_size; i++)
	{
		bias_out[i] = ih_weight_out[i] + ih_bias[i] + hh_weight_out[i] + hh_bias[i];
	}

	free(ih_weight_out);
	free(ih_bias);
	free(hh_weight_out);
	free(hh_bias);
	ih_weight_out = NULL;
	ih_bias = NULL;
	hh_weight_out = NULL;
	hh_bias = NULL;
	return bias_out;
}


float* lw_fc_bais_add(lw_head_t* head, float* fc_weight_out, float* fc_bias)
{
	float* bias_out = (float*)calloc((uint32_t)(sizeof(float) * head->classfication), 1); 

	for (uint32_t i = 0; i < (uint32_t)head->classfication; i++)
	{
		bias_out[i] = fc_weight_out[i] + fc_bias[i];
	}

	free(fc_weight_out);
	free(fc_bias);
	fc_weight_out = NULL;
	fc_bias = NULL;
	return bias_out;
}


float* lw_tanh_active(lw_head_t* head, float* bias_out, lw_layer_t layer)
{
	float* tanh_out = (float*)calloc((uint32_t)(sizeof(float) * head->hidden_size), 1); 

	for (uint32_t i = 0; i < (uint32_t)head->hidden_size; i++)
	{
		tanh_out[i] = (float)tanhl(bias_out[i]);
	}

	free(bias_out);
	bias_out = NULL;
	return tanh_out;
}



lw_head_t* lw_read_head()
{
	uint32_t len = sizeof(lw_head_t);
	uint32_t seek = 0;

	lw_head_t* head = (lw_head_t*)lw_read_data(len, seek);

	return head;
}


float* lw_read_sample(lw_head_t* head)
{
	uint32_t len = sizeof(float) * (uint32_t)head->seq_len;
	uint32_t seek = sizeof(float) * (uint32_t)(head->head_len + head->para_size);

	float* sample = lw_read_data(len, seek);

	return sample;
}


float* lw_read_data(uint32_t len, uint32_t seek)
{
	
	#ifdef ON_CPU
	{
		FILE* file_read = NULL;
		float* buffer = (float*)calloc(len, 1); 

		fopen_s(&file_read, (char*)FILE_NAME, "rb");

		fseek(file_read, seek, SEEK_SET);

		fread(buffer, len, 1, file_read); 
		fclose(file_read); 

		return buffer;
	}
	#endif

	
	#ifdef ON_MCU
	{
		float* buffer = (float*)calloc(len, 1); 
		uint32_t flash_addr = (uint32_t)FLASH_DATA_BASE_ADDR + seek;
		for (uint32_t i = 0; i < len/sizeof(float); i++)
		{
			buffer[i] = *(float*)(flash_addr);
			flash_addr += 4; 
		}

		return buffer;
	}
	#endif
}





