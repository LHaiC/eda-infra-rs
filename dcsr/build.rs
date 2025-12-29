fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=csrc");

        let cuda_path = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());

        let cuda_include = format!("{}/include", cuda_path);
        let csrc_headers = ucc::import_csrc();

        let mut cl_cuda = ucc::cl_cuda();
        cl_cuda.include(&csrc_headers);
        cl_cuda.include(&cuda_include);
        cl_cuda.debug(false).opt_level(3);
        cl_cuda.file("csrc/utils/cuda-utils.cu")
            .file("csrc/test_helpers.cu");
        cl_cuda.compile("dcsr_test_helpers");

        println!("cargo:rustc-link-lib=static=dcsr_test_helpers");
        println!("cargo:rustc-link-lib=static=ulibcu");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}