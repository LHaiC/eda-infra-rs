//! this build script exports the csrc dir to dependents.

fn main() {
    println!("Building the C source files of dcsr..");
    println!("cargo:rerun-if-changed=csrc");

    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    let cuda_include = format!("{}/include", cuda_path);
    let csrc_headers = ucc::import_csrc();
    let mut cl_cpp = ucc::cl_cpp_openmp();
    cl_cpp.cpp(true)
        .include(&csrc_headers)
        .include(&cuda_include);
    cl_cpp.debug(false).opt_level(3);
    cl_cpp.file("csrc/dcsr/dcsr.cpp");
    cl_cpp.compile("dcsrc");
    println!("cargo:rustc-link-lib=static=dcsrc");

    #[cfg(feature = "cuda")]
    let cl_cuda = {
        let mut cl_cuda = ucc::cl_cuda();
        cl_cuda.include(&csrc_headers);
        cl_cuda.debug(false).opt_level(3);
        cl_cuda.file("csrc/utils/cuda-utils.cu")
            .file("csrc/dcsr/compact.cu")
            .file("csrc/test_helpers.cu"); 
        cl_cuda.compile("dcsr_cuda");
        println!("cargo:rustc-link-lib=static=ulibcu");
        println!("cargo:rustc-link-lib=dylib=cudart");
        cl_cuda
    };

    ucc::bindgen([
        "csrc/dcsr/dcsr_capi.h"
    ], "dcsr.rs");

    ucc::export_csrc();
    ucc::make_compile_commands(&[
        &cl_cpp,
        #[cfg(feature = "cuda")] &cl_cuda
    ]);
}
