use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};
use glam::{Mat4, Vec3};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
]);

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

pub struct CameraController {
    speed: f32,
    sensitivity: f32,
    yaw: f32,
    pitch: f32,

    // Key states
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            yaw: -90.0_f32.to_radians(), // Start looking "forward" along -Z
            pitch: 0.0,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
        }
    }

    pub fn look_at(&mut self, eye: Vec3, target: Vec3) {
        let dir = target - eye;
        if dir.length_squared() < 1e-12 {
            return;
        }

        let forward = dir.normalize();

        // Match update_camera()'s forward vector convention:
        // forward = (cos_pitch*cos_yaw, sin_pitch, cos_pitch*sin_yaw)
        self.yaw = forward.z.atan2(forward.x);
        self.pitch = forward.y.asin();

        // Clamp pitch so you don't flip over
        self.pitch = self
            .pitch
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    KeyCode::Space => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    KeyCode::ShiftLeft => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.yaw += (mouse_dx as f32) * self.sensitivity;
        self.pitch -= (mouse_dy as f32) * self.sensitivity;

        // Clamp pitch so you don't flip over
        self.pitch = self
            .pitch
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        // Calculate forward vector from yaw/pitch
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        let forward = Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();

        let right = forward.cross(Vec3::Y).normalize();
        let up = Vec3::Y;

        // Move the camera
        if self.is_forward_pressed {
            camera.eye += forward * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward * self.speed;
        }
        if self.is_right_pressed {
            camera.eye += right * self.speed;
        }
        if self.is_left_pressed {
            camera.eye -= right * self.speed;
        }
        if self.is_up_pressed {
            camera.eye += up * self.speed;
        }
        if self.is_down_pressed {
            camera.eye -= up * self.speed;
        }

        // Update target
        camera.target = camera.eye + forward;
    }
}
