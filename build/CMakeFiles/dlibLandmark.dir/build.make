# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fa/Desktop/Data/DlibLandlandmark

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fa/Desktop/Data/DlibLandlandmark/build

# Include any dependencies generated for this target.
include CMakeFiles/dlibLandmark.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dlibLandmark.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dlibLandmark.dir/flags.make

CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o: CMakeFiles/dlibLandmark.dir/flags.make
CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o: ../dlibLandmark.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fa/Desktop/Data/DlibLandlandmark/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o -c /home/fa/Desktop/Data/DlibLandlandmark/dlibLandmark.cpp

CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fa/Desktop/Data/DlibLandlandmark/dlibLandmark.cpp > CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.i

CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fa/Desktop/Data/DlibLandlandmark/dlibLandmark.cpp -o CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.s

CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o.requires:

.PHONY : CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o.requires

CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o.provides: CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o.requires
	$(MAKE) -f CMakeFiles/dlibLandmark.dir/build.make CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o.provides.build
.PHONY : CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o.provides

CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o.provides.build: CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o


# Object files for target dlibLandmark
dlibLandmark_OBJECTS = \
"CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o"

# External object files for target dlibLandmark
dlibLandmark_EXTERNAL_OBJECTS =

dlibLandmark: CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o
dlibLandmark: CMakeFiles/dlibLandmark.dir/build.make
dlibLandmark: /usr/local/lib/libopencv_core.so
dlibLandmark: /usr/local/lib/libopencv_highgui.so
dlibLandmark: /usr/local/lib/libopencv_imgcodecs.so
dlibLandmark: /usr/local/lib/libopencv_imgproc.so
dlibLandmark: /usr/local/lib/libopencv_ml.so
dlibLandmark: /usr/local/lib/libopencv_objdetect.so
dlibLandmark: /usr/local/lib/libopencv_photo.so
dlibLandmark: /usr/local/lib/libopencv_shape.so
dlibLandmark: /usr/local/lib/libopencv_stitching.so
dlibLandmark: /usr/local/lib/libopencv_superres.so
dlibLandmark: /usr/local/lib/libopencv_video.so
dlibLandmark: /usr/local/lib/libopencv_videoio.so
dlibLandmark: /usr/local/lib/libopencv_videostab.so
dlibLandmark: /usr/local/lib/libopencv_tracking.so
dlibLandmark: /usr/local/lib/libopencv_calib3d.so
dlibLandmark: /usr/local/lib/libdlib.so
dlibLandmark: /usr/local/lib/libopenblas.so
dlibLandmark: /usr/local/lib/liblapack.so
dlibLandmark: CMakeFiles/dlibLandmark.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fa/Desktop/Data/DlibLandlandmark/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dlibLandmark"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dlibLandmark.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dlibLandmark.dir/build: dlibLandmark

.PHONY : CMakeFiles/dlibLandmark.dir/build

CMakeFiles/dlibLandmark.dir/requires: CMakeFiles/dlibLandmark.dir/dlibLandmark.cpp.o.requires

.PHONY : CMakeFiles/dlibLandmark.dir/requires

CMakeFiles/dlibLandmark.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dlibLandmark.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dlibLandmark.dir/clean

CMakeFiles/dlibLandmark.dir/depend:
	cd /home/fa/Desktop/Data/DlibLandlandmark/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fa/Desktop/Data/DlibLandlandmark /home/fa/Desktop/Data/DlibLandlandmark /home/fa/Desktop/Data/DlibLandlandmark/build /home/fa/Desktop/Data/DlibLandlandmark/build /home/fa/Desktop/Data/DlibLandlandmark/build/CMakeFiles/dlibLandmark.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dlibLandmark.dir/depend

