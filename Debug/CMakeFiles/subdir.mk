################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CXX_SRCS += \
../CMakeFiles/feature_tests.cxx 

C_SRCS += \
../CMakeFiles/feature_tests.c 

OBJS += \
./CMakeFiles/feature_tests.o 

CXX_DEPS += \
./CMakeFiles/feature_tests.d 

C_DEPS += \
./CMakeFiles/feature_tests.d 


# Each subdirectory must supply rules for building sources it contributes
CMakeFiles/%.o: ../CMakeFiles/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

CMakeFiles/%.o: ../CMakeFiles/%.cxx
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


