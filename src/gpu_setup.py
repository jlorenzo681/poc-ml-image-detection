import sys
import os
import site

def setup_tf_cuda():
    # Only try this if "site-packages" has Nvidia libraries
    try:
        site_packages = site.getsitepackages()
    except Exception:
        return
        
    if not site_packages:
        return
        
    nvidia_base = os.path.join(site_packages[0], "nvidia")
    if not os.path.exists(nvidia_base):
        return
        
    paths = []
    for d in os.listdir(nvidia_base):
        lib_dir = os.path.join(nvidia_base, d, "lib")
        if os.path.exists(lib_dir):
            paths.append(lib_dir)
    
    if paths:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        # Check if our paths are already in LD_LIBRARY_PATH to avoid infinite loops
        if all(p in current for p in paths):
            return
            
        new_path = ":".join(paths)
        if current:
            new_path += ":" + current
            
        os.environ["LD_LIBRARY_PATH"] = new_path
        
        # We must re-execute the python process because glibc ld.so reads LD_LIBRARY_PATH exactly once AT STARTUP.
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Running this upon import makes sure it happens before TensorFlow is ever imported.
setup_tf_cuda()
