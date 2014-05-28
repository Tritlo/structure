#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define MAXNUMBEROFKERNELS 1
#define MAX_SOURCE_SIZE (0x100000)
#define USEGPU 1
#define UPDATEZLOCIKERNEL 0


typedef struct CLDict {
    int numProgramsInDict;
    char **ProgramNames;
    cl_program *Programs;    
    cl_platform_id platform_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_context context;
    cl_device_id device_id; 
    cl_command_queue commands;
} CLDict;

int InitCLDict(CLDict *clDictToInit){
    char *ProgramNames[MAXNUMBEROFKERNELS];
    cl_program Programs[MAXNUMBEROFKERNELS];    
    cl_platform_id platform_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    cl_context context;
    cl_device_id device_id; 
    cl_command_queue commands;
    int DEVICETYPE;
    int err;
    DEVICETYPE =  USEGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    err = clGetDeviceIDs(platform_id, DEVICETYPE, 1, &device_id, &ret_num_devices);
    if (err != CL_SUCCESS)
    {
        printf("retval %d\n",(int) ret);
    switch(err){
    case CL_INVALID_PLATFORM:
    	printf("invalid platform!");
                break;
    case CL_INVALID_VALUE:
    	printf("invalid value");
                break;
    case CL_DEVICE_NOT_FOUND:
    	printf("device not found");
                break;
    case CL_INVALID_DEVICE_TYPE:
            if(USEGPU){
    	    printf("invalid device: GPU\n");
    	} else {
    	    printf("invalid device: CPU\n");
    	}
                break;
    }
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }




    clDictToInit->numProgramsInDict = 0;
    clDictToInit->ProgramNames = ProgramNames;
    clDictToInit->Programs = Programs;
    clDictToInit->platform_id = platform_id;
    clDictToInit->ret_num_devices = ret_num_devices;
    clDictToInit->ret_num_platforms = ret_num_platforms;
    clDictToInit->device_id = device_id;
    clDictToInit->context = context;
    clDictToInit->commands = commands;
    return EXIT_SUCCESS;
}

void ReleaseCLDict(CLDict *clDict){
    int i;
    for(i = 0; i < clDict->numProgramsInDict; i++){
        clReleaseProgram(clDict->Programs[i]);
    } 
    clReleaseCommandQueue(clDict->commands);
    clReleaseContext(clDict->context);
}

/*
 * If we've already compiled this program,
 * return the index in CompiledPrograms in which it is located.
 * to save on speed, we should really just keep track of in the code
 * where each program is and use that, i.e. define a constant for the
 * location of each program.
 */
int alreadyCompiled(char * programFilename, CLDict * clDict){
    int i, comp;
    for(i = 0; i < clDict->numProgramsInDict && i < MAXNUMBEROFKERNELS; i++){
      comp = strcmp(programFilename,clDict->ProgramNames[i]); 
      if(comp == 0){
        return i;     
      }
    }
    return -1;
}


int CompileProgram(char * programFilename, CLDict *clDict){
    int indIfAlreadyCompiled;
    FILE *fp;
    char *KernelSource; 
    size_t source_size;
    
    int err;
    int indOfProgram;
    /*cl_int ret;*/


    cl_program program;
    indIfAlreadyCompiled = alreadyCompiled(programFilename, clDict);
    if(indIfAlreadyCompiled >= 0){
        return indIfAlreadyCompiled;        
    }

    /* Load the source code containing the kernel*/
    fp = fopen(programFilename, "r");
    if (!fp) {
    	fprintf(stderr, "Failed to load kernel %s.\n", programFilename);
        exit(EXIT_FAILURE);
    }

    KernelSource = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(KernelSource, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    program = clCreateProgramWithSource(clDict->context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        printf("Size: %d\n", (int)source_size);
        exit(EXIT_FAILURE);
    }


    free(KernelSource);


    err = clBuildProgram(program, 1, &clDict->device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, clDict->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(EXIT_FAILURE);
    }

    indOfProgram = clDict->numProgramsInDict++;
    clDict->ProgramNames[indOfProgram] = programFilename;
    clDict->Programs[indOfProgram] = program;
    return indOfProgram;
}

/*
int main(int argc, char *argv[]){
    CLDict clDict;
    int a,b;
    InitCLDict(&clDict);
    a = CompileProgram("UpdateZLoci.cl", &clDict); 
    b = CompileProgram("UpdateZLoci.cl", &clDict); 
    ReleaseCLDict(&clDict);
    printf("Compiled!\n");
    printf("a %d, b %d\n",a,b);
    return EXIT_SUCCESS;
}*/
