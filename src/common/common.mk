NVCC        := nvcc
CXX         := g++
CFLAGS      := -O2 -std=c++11
CUFLAGS     := -O2 --compiler-options '-fPIC'

INCLUDES    := -I/usr/local/cuda/include
LIBS        :=

OBJDIR      := obj
OBJS        := $(patsubst %.cpp, $(OBJDIR)/%.o, $(CCFILES)) \
	                      $(patsubst %.cu,  $(OBJDIR)/%.o, $(CUFILES))

$(OBJDIR)/%.o: %.cpp
		@mkdir -p $(dir $@)
			$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.o: %.cu
		@mkdir -p $(dir $@)
			$(NVCC) $(CUFLAGS) $(INCLUDES) -c $< -o $@

$(EXECUTABLE): $(OBJS)
		@mkdir -p $(dir $@)
		$(NVCC) $(CUFLAGS) $(OBJS) -o $@ $(LIBS)

clean:
		rm -rf $(OBJDIR) $(EXECUTABLE)
