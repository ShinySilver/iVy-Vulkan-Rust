use glam::Vec3;
use num_enum::FromPrimitive;

#[repr(transparent)]
#[derive(Default, Debug, Copy, Clone)]
pub struct Voxel(u16);

#[repr(u16)]
#[derive(Debug, PartialEq, Copy, Clone, FromPrimitive)]
pub enum Material {
    #[num_enum(default)]
    Air,
    DebugRed,
    DebugGreen,
    DebugBlue,
    Stone,
    Dirt,
    Grass,
    OakLog,
    OakLeaves,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum MaterialCategory {
    Debug,
    Terrain,
    Leaves,
    Logs,
}

impl Voxel {
    fn new(material: Material) -> Self { Self { 0: material as u16 & 0x7fu16 } }
    pub fn material(&self) -> Material { Material::try_from(self.0 & 0x7fu16).unwrap() }
    pub fn encoded_normal(&self) -> u16 { (self.0 & !0x7fu16) >> 7 }
    pub fn normal(&self) -> Vec3 { Default::default() }
    pub fn set_normal(&mut self, normal: Vec3) {
        let map = |v: f32| (((v + 1.0) * 3.5).round() as u16).min(0b111u16);
        self.0 = self.0 & 0x7fu16 | (map(normal.x) << 13) | (map(normal.y) << 10) | (map(normal.z) << 7);
    }
}

impl PartialEq for Voxel {
    fn eq(&self, other: &Self) -> bool { self.0 & 0x7fu16 == other.0 & 0x7fu16 }
}

impl From<Material> for Voxel {
    fn from(m: Material) -> Self { Voxel::new(m) }
}

impl From<Voxel> for u16 {
    fn from(v: Voxel) -> Self { v.0 }
}

impl Material {
    pub fn category(&self) -> MaterialCategory {
        match self {
            Material::Air | Material::DebugRed | Material::DebugGreen | Material::DebugBlue => MaterialCategory::Debug,
            Material::Stone | Material::Dirt | Material::Grass => MaterialCategory::Terrain,
            Material::OakLog => MaterialCategory::Logs,
            Material::OakLeaves => MaterialCategory::Leaves,
        }
    }
}