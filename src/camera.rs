#[repr(C)]
#[derive(Copy, Clone)]
pub struct CameraUniforms {
    pub screen_size: glam::UVec2,
    pub camera_position: glam::Vec3,
    pub _padding0: u32, // pad to 16 bytes
    pub view_matrix: glam::Mat4,
    pub projection_matrix: glam::Mat4,
    pub world_width: u32,
    pub _padding1: [u32; 3], // pad to 16 bytes
}