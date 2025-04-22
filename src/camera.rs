use glam;
use winit::keyboard::KeyCode;
use winit_input_helper::WinitInputHelper;

pub struct Camera {
    pub position: glam::Vec3,
    pub forward: glam::Vec3,
    pub up: glam::Vec3,
    matrix: glam::Mat4,
}

impl Camera {
    pub fn new(position: glam::Vec3, forward: glam::Vec3, up: glam::Vec3) -> Self {
        Self {
            position,
            forward,
            up,
            matrix: glam::Mat4::IDENTITY,
        }
    }
    pub fn view_matrix(&self) -> glam::Mat4 {
        self.matrix
    }
    pub fn update(&mut self, input_helper: &WinitInputHelper) {
        let right = self.forward.cross(self.up).normalize();
        let speed = 0.1;
        if input_helper.key_held(KeyCode::KeyW) {
            self.position += self.forward * speed;
        }
        if input_helper.key_held(KeyCode::KeyS) {
            self.position -= self.forward * speed;
        }
        if input_helper.key_held(KeyCode::KeyA) {
            self.position -= right * speed;
        }
        if input_helper.key_held(KeyCode::KeyD) {
            self.position += right * speed;
        }
        if input_helper.key_held(KeyCode::Space) {
            self.position += self.up * speed;
        }
        if input_helper.held_shift() {
            self.position -= self.up * speed;
        }
        self.matrix = glam::Mat4::look_at_rh(self.position, self.position + self.forward, self.up);
    }
}

pub struct Projection {
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
    matrix: glam::Mat4,
}

impl Projection {
    pub fn new(fov_y_radians: f32, near: f32, far: f32) -> Self {
        Self {
            fov_y_radians,
            near,
            far,
            matrix: glam::Mat4::IDENTITY,
        }
    }
    pub fn projection_matrix(&self) -> glam::Mat4 {
        self.matrix
    }
    pub fn update(&mut self, input_helper: &WinitInputHelper) {
        let resolution = input_helper.resolution().unwrap();
        let aspect_ratio = resolution.0 as f32 / resolution.1 as f32;
        self.matrix =
            glam::Mat4::perspective_rh(self.fov_y_radians, aspect_ratio, self.near, self.far);
    }
}
