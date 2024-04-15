1. We have implemented RNN to run on STM32F405 and STM32FH723 in a low-memory manner. The code provided here is for STM32FH723, but it can be easily ported to STM32F405 with minimal changes, except for UART initialization. If you need assistance with porting it to STM32F405, please leave a message.

2. This project can be directly run on MDK Keil for STM32FH723, but it requires rebuilding the project for STM32F405.

3. Why are we releasing the code for STM32FH723 instead of STM32F405? This is because we are preparing to upgrade to a higher-performance MCU for use in wireless sensor network nodes, meeting the requirements of edge computing.

4. The provided code does not include application-specific code for wireless sensor network (WSN) applications due to confidentiality concerns.

5. For any other related interests, please feel free to leave a message.
