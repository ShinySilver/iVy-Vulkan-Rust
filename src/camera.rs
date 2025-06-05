use std::time::Instant;
use glam;
use glam::vec2;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;
use winit::window::{CursorGrabMode, Window};
use winit_input_helper::WinitInputHelper;

pub struct Camera {
    pub position: glam::Vec3,
    pub forward: glam::Vec3,
    pub up: glam::Vec3,
    pub time: f32,
    matrix: glam::Mat4,
    mouse_delta_smoothed: glam::Vec2,
    is_cursor_locked: bool,
    start_time: Instant,
}

impl Camera {
    pub fn new(position: glam::Vec3, forward: glam::Vec3, up: glam::Vec3) -> Self {
        Self {
            position,
            forward,
            up,
            time: 0.0,
            matrix: glam::Mat4::IDENTITY,
            mouse_delta_smoothed: Default::default(),
            is_cursor_locked: false,
            start_time: Instant::now(),
        }
    }
    pub fn view_matrix(&self) -> glam::Mat4 {
        self.matrix
    }
    pub fn update(&mut self, window: &Window, input_helper: &WinitInputHelper) {
        self.time = self.start_time.elapsed().as_secs_f32();
        let right = self.forward.cross(self.up).normalize();
        let speed = if input_helper.held_shift() { 10.0 } else { 1.0 };

        // WASD movement
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

        // Up/Down
        if input_helper.key_held(KeyCode::Space) {
            self.position += self.up * speed;
        }
        if input_helper.key_held(KeyCode::ControlLeft) {
            self.position -= self.up * speed;
        }

        // Mouse lock / unlock
        if input_helper.mouse_pressed(MouseButton::Left) {
            window.set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_e| window.set_cursor_grab(CursorGrabMode::Locked)).unwrap();
            window.set_cursor_visible(false);
            self.is_cursor_locked = true;
        }
        if input_helper.held_alt() && self.is_cursor_locked {
            window.set_cursor_grab(CursorGrabMode::None).unwrap();
            window.set_cursor_visible(true);
            self.is_cursor_locked = false;
        }

        // Mouse movement
        if self.is_cursor_locked {
            let (raw_dx, raw_dy) = input_helper.mouse_diff();
            let smoothing = 0.5;
            self.mouse_delta_smoothed = self.mouse_delta_smoothed.lerp(vec2(raw_dx, raw_dy), smoothing);

            let delta_x = self.mouse_delta_smoothed.x;
            let delta_y = self.mouse_delta_smoothed.y;

            if delta_x != 0f32 || delta_y != 0f32 {
                let sensitivity = 0.002;

                // Rotate horizontally around the global Y axis (yaw)
                let yaw = glam::Quat::from_axis_angle(self.up, -delta_x as f32 * sensitivity);
                self.forward = (yaw * self.forward).normalize();

                // Rotate vertically around the right axis (pitch)
                let right = self.forward.cross(self.up).normalize();
                let pitch = glam::Quat::from_axis_angle(right, -delta_y as f32 * sensitivity);

                let new_forward = (pitch * self.forward).normalize();

                // Limit pitch to prevent flipping
                let dot = new_forward.dot(self.up);
                if dot.abs() < 0.99 {
                    self.forward = new_forward;
                }
            }
        }

        // Creating the view matrix
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
