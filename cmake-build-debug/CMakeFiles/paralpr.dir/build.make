# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/gh0stadian/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/202.7660.37/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/gh0stadian/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/202.7660.37/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gh0stadian/CLionProjects/paralpr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gh0stadian/CLionProjects/paralpr/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/paralpr.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/paralpr.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/paralpr.dir/flags.make

CMakeFiles/paralpr.dir/main.cpp.o: CMakeFiles/paralpr.dir/flags.make
CMakeFiles/paralpr.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gh0stadian/CLionProjects/paralpr/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/paralpr.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/paralpr.dir/main.cpp.o -c /home/gh0stadian/CLionProjects/paralpr/main.cpp

CMakeFiles/paralpr.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/paralpr.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gh0stadian/CLionProjects/paralpr/main.cpp > CMakeFiles/paralpr.dir/main.cpp.i

CMakeFiles/paralpr.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/paralpr.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gh0stadian/CLionProjects/paralpr/main.cpp -o CMakeFiles/paralpr.dir/main.cpp.s

# Object files for target paralpr
paralpr_OBJECTS = \
"CMakeFiles/paralpr.dir/main.cpp.o"

# External object files for target paralpr
paralpr_EXTERNAL_OBJECTS =

paralpr: CMakeFiles/paralpr.dir/main.cpp.o
paralpr: CMakeFiles/paralpr.dir/build.make
paralpr: /usr/lib/x86_64-linux-gnu/libpng.so
paralpr: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
paralpr: /usr/lib/x86_64-linux-gnu/libpthread.so
paralpr: /usr/lib/x86_64-linux-gnu/libz.so
paralpr: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
paralpr: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
paralpr: CMakeFiles/paralpr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gh0stadian/CLionProjects/paralpr/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable paralpr"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/paralpr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/paralpr.dir/build: paralpr

.PHONY : CMakeFiles/paralpr.dir/build

CMakeFiles/paralpr.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/paralpr.dir/cmake_clean.cmake
.PHONY : CMakeFiles/paralpr.dir/clean

CMakeFiles/paralpr.dir/depend:
	cd /home/gh0stadian/CLionProjects/paralpr/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gh0stadian/CLionProjects/paralpr /home/gh0stadian/CLionProjects/paralpr /home/gh0stadian/CLionProjects/paralpr/cmake-build-debug /home/gh0stadian/CLionProjects/paralpr/cmake-build-debug /home/gh0stadian/CLionProjects/paralpr/cmake-build-debug/CMakeFiles/paralpr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/paralpr.dir/depend

