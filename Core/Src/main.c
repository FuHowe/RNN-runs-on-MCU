#include "lw_rnn.h"

int main(void)
{
#ifdef ON_MCU
	lw_init_borad();
#endif	

	float* model_out = lw_model(); 

	printf("model out = [%f %f %f %f]\r\n", model_out[0], model_out[1], model_out[2], model_out[3]);

	free(model_out);

	return 0;
}





