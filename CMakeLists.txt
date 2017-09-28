cmake_minimum_required (VERSION 2.6)
 
 
set (CMAKE_CXX_STANDARD 11)

add_compile_options(-std=c++11)

project (dliblandmark)
 
INCLUDE_DIRECTORIES(
/usr/local/include
/usr/include/boost
)

link_libraries(
/usr/local/lib/libopencv_core.so 
/usr/local/lib/libopencv_highgui.so 
/usr/local/lib/libopencv_imgcodecs.so 
/usr/local/lib/libopencv_imgproc.so 
/usr/local/lib/libopencv_ml.so 
/usr/local/lib/libopencv_objdetect.so 
/usr/local/lib/libopencv_photo.so 
/usr/local/lib/libopencv_shape.so 
/usr/local/lib/libopencv_stitching.so 
/usr/local/lib/libopencv_superres.so 
/usr/local/lib/libopencv_video.so 
/usr/local/lib/libopencv_videoio.so 
/usr/local/lib/libopencv_videostab.so 
/usr/local/lib/libopencv_tracking.so
/usr/local/lib/libopencv_calib3d.so
/usr/local/lib/libdlib.so
/usr/local/lib/libopenblas.so
/usr/local/lib/liblapack.so
)

 SET(SRC_LIST dlibLandmark.cpp
        )
		
		
		
ADD_EXECUTABLE(dlibLandmark ${SRC_LIST})
