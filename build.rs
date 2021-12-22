#[cfg(feature = "gpu")]
extern crate cmake;

#[cfg(feature = "gpu")]
use cmake::Config;

fn main() {
    #[cfg(feature = "gpu")]
    {
        let dst = Config::new("damavand-gpu").build();
        println!("cargo:rustc-link-search=native={}", dst.display());
        // println!("cargo:rustc-link-search=native={}", "./_skbuild/linux-x86_64-3.8/cmake-install/damavand");
        println!("cargo:rustc-link-lib=dylib=damavand-gpu");
    }
}
