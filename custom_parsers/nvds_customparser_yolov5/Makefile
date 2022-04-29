CC:= g++ -std=c++17
NVCC:=/usr/local/cuda/bin/nvcc

CFLAGS:= -Wall -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream-5.1/sources/includes -I/usr/local/cuda/include

LIBS:= -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h)
SRCFILES:= nvds_parsebbox_yolov5.cpp \
           yololayer.cu

TARGET_LIB:= libnvds_infercustomparser_yolov5.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
	rm -rf $(TARGET_OBJS)
