Import("env")
import os

# Add the missing target directory source files for esp32-camera library
lib_deps_dir = env.subst("$PROJECT_LIBDEPS_DIR/$PIOENV")
esp32_camera_dir = os.path.join(lib_deps_dir, "esp32-camera")

if os.path.exists(esp32_camera_dir):
    # Add target directory to build
    target_dir = os.path.join(esp32_camera_dir, "target")
    
    # For ESP32, add target/esp32/ll_cam.c and target/xclk.c
    env.Append(
        CPPPATH=[
            os.path.join(esp32_camera_dir, "target", "private_include"),
            os.path.join(esp32_camera_dir, "target", "jpeg_include"),
        ]
    )
    
    print("ESP32-Camera: Added target directory to build")
