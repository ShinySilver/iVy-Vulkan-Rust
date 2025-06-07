use std::{env, process};
use std::fs;
use std::path::{Path, PathBuf};
use spirv_tools::opt::Optimizer;
use walkdir::WalkDir;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("shaders");

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.set_warnings_as_errors();

    for entry in WalkDir::new(shader_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().is_file())
    {
        let path = entry.path();
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let shader_kind = match extension {
            "vert" => shaderc::ShaderKind::Vertex,
            "frag" => shaderc::ShaderKind::Fragment,
            "comp" => shaderc::ShaderKind::Compute,
            "geom" => shaderc::ShaderKind::Geometry,
            "tesc" => shaderc::ShaderKind::TessControl,
            "tese" => shaderc::ShaderKind::TessEvaluation,
            _ => continue,
        };

        let source = fs::read_to_string(path).expect("Failed to read shader");
        let filename = path.file_name().unwrap().to_str().unwrap();

        // 1) compile GLSL/HLSL → SPIR-V
        let compiled = compiler
            .compile_into_spirv(&source, shader_kind, filename, "main", Some(&options))
            .unwrap_or_else(|e| {
                eprintln!("Could not compile {filename}: {e}");
                process::exit(1);
            });

        // 2) SPIR-V → optimized SPIR-V
        let mut optimizer = spirv_tools::opt::compiled::CompiledOptimizer::default();
        optimizer.register_performance_passes();
        let optimized_binary = optimizer.optimize(compiled.as_binary(), &mut |msg| {
            eprintln!("[compiled] optimizer message: {:#?}", msg);
        }, None).expect("Could not optimize shader.");

        // 3) write out the optimized shader
        let spv_path = out_dir.join(format!("{filename}.spv"));
        fs::write(&spv_path, optimized_binary.as_bytes()).expect("Failed to write optimized SPIR-V");
        //fs::write(&spv_path, compiled.as_binary_u8()).expect("Failed to write optimized SPIR-V");

        println!("cargo:rerun-if-changed={}", path.display());
    }
}
