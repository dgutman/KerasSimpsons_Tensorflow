Go into the app directory and type runCPUTest.sh or runGPUTest.sh

It's actually the same script but I swagged cpu/and gpu... it should in both
cases be using the same docker container

The last parameter in the script either 10000 or 15000 refers to how big of a matrix
it is running its test on... if I make this TOO big I will either run out of GPU or
CPU memory depending.  On whorlwind, which has 128GB of RAM you will crash the GPU version
much sooner since the GPU's have <12GB of RAM
