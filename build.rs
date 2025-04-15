use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("shaders");

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

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
            _ => continue, // skip unsupported files
        };

        let source = fs::read_to_string(path).expect("Failed to read shader");
        let filename = path.file_name().unwrap().to_str().unwrap();

        let compiled_result = compiler
            .compile_into_spirv(&source, shader_kind, filename, "main", Some(&options))
            .expect(&format!("Failed to compile shader: {}", filename));

        let spv_path = out_dir.join(format!("{}.spv", filename));
        fs::write(&spv_path, compiled_result.as_binary_u8())
            .expect("Failed to write compiled SPIR-V");
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
