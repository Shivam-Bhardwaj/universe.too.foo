//! Window management and input handling

use crate::renderer::Renderer;
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, ElementState, MouseButton, DeviceEvent},
    event_loop::{EventLoop, ActiveEventLoop, ControlFlow},
    window::{Window, WindowId, CursorGrabMode},
    keyboard::{KeyCode, PhysicalKey},
    dpi::PhysicalSize,
};
use std::sync::Arc;
use std::collections::HashSet;
use std::time::Instant;

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    universe_dir: std::path::PathBuf,

    // Input state
    keys_pressed: HashSet<KeyCode>,
    mouse_captured: bool,

    // Timing
    last_frame: Instant,
}

impl App {
    pub fn new(universe_dir: std::path::PathBuf) -> Self {
        Self {
            window: None,
            renderer: None,
            universe_dir,
            keys_pressed: HashSet::new(),
            mouse_captured: false,
            last_frame: Instant::now(),
        }
    }

    fn handle_input(&mut self, dt: f32) {
        if let Some(renderer) = &mut self.renderer {
            let mut forward = 0.0;
            let mut right = 0.0;
            let mut up = 0.0;

            if self.keys_pressed.contains(&KeyCode::KeyW) { forward += 1.0; }
            if self.keys_pressed.contains(&KeyCode::KeyS) { forward -= 1.0; }
            if self.keys_pressed.contains(&KeyCode::KeyD) { right += 1.0; }
            if self.keys_pressed.contains(&KeyCode::KeyA) { right -= 1.0; }
            if self.keys_pressed.contains(&KeyCode::Space) { up += 1.0; }
            if self.keys_pressed.contains(&KeyCode::ShiftLeft) { up -= 1.0; }

            // Speed modifiers
            let speed_mult = if self.keys_pressed.contains(&KeyCode::KeyQ) { 0.1 }
                           else if self.keys_pressed.contains(&KeyCode::KeyE) { 10.0 }
                           else { 1.0 };

            let orig_speed = renderer.camera.speed;
            renderer.camera.speed *= speed_mult;
            renderer.camera.translate(forward, right, up, dt as f64);
            renderer.camera.speed = orig_speed;

            // Time controls
            if self.keys_pressed.contains(&KeyCode::Comma) {
                renderer.time_controller.set_rate(renderer.time_controller.rate() * 0.5);
            }
            if self.keys_pressed.contains(&KeyCode::Period) {
                renderer.time_controller.set_rate(renderer.time_controller.rate() * 2.0);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("Universe - Universe Visualization")
            .with_inner_size(PhysicalSize::new(1920, 1080));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let renderer = pollster::block_on(
            Renderer::new(Arc::clone(&window), &self.universe_dir)
        ).expect("Failed to create renderer");

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size);
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.keys_pressed.insert(key);

                            // Special keys
                            match key {
                                KeyCode::Escape => {
                                    if self.mouse_captured {
                                        self.mouse_captured = false;
                                        if let Some(window) = &self.window {
                                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                                            window.set_cursor_visible(true);
                                        }
                                    } else {
                                        event_loop.exit();
                                    }
                                }
                                KeyCode::KeyP => {
                                    if let Some(r) = &mut self.renderer {
                                        r.time_controller.toggle_pause();
                                    }
                                }
                                KeyCode::Digit1 => {
                                    // Go to Earth
                                    if let Some(r) = &mut self.renderer {
                                        let earth = r.solar_system.body_position(universe_sim::Body::Earth);
                                        r.camera.set_position(glam::DVec3::new(earth.x, earth.y, earth.z + 1e8));
                                        r.camera.look_at(glam::DVec3::new(earth.x, earth.y, earth.z));
                                    }
                                }
                                KeyCode::Digit2 => {
                                    // Go to Mars
                                    if let Some(r) = &mut self.renderer {
                                        let mars = r.solar_system.body_position(universe_sim::Body::Mars);
                                        r.camera.set_position(glam::DVec3::new(mars.x, mars.y, mars.z + 1e8));
                                        r.camera.look_at(glam::DVec3::new(mars.x, mars.y, mars.z));
                                    }
                                }
                                KeyCode::Digit0 => {
                                    // Go to Sun overview
                                    if let Some(r) = &mut self.renderer {
                                        r.camera.set_position(glam::DVec3::new(0.0, 5e11, 0.0));
                                        r.camera.look_at(glam::DVec3::ZERO);
                                    }
                                }
                                KeyCode::KeyT => {
                                    // Toggle rendering pipeline
                                    if let Some(r) = &mut self.renderer {
                                        r.toggle_pipeline();
                                    }
                                }
                                _ => {}
                            }
                        }
                        ElementState::Released => {
                            self.keys_pressed.remove(&key);
                        }
                    }
                }
            }

            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !self.mouse_captured {
                    self.mouse_captured = true;
                    if let Some(window) = &self.window {
                        let _ = window.set_cursor_grab(CursorGrabMode::Locked)
                            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                        window.set_cursor_visible(false);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                self.handle_input(dt);

                if let Some(renderer) = &mut self.renderer {
                    renderer.update(dt);
                    let _ = renderer.prepare_frame();
                    let _ = renderer.render();

                    // Update title with info
                    if let Some(window) = &self.window {
                        window.set_title(&format!("Universe | {}", renderer.get_info()));
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if self.mouse_captured {
                if let Some(renderer) = &mut self.renderer {
                    renderer.camera.rotate(dx as f32, dy as f32);
                }
            }
        }
    }
}

/// Run the windowed renderer
pub fn run(universe_dir: &std::path::Path) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(universe_dir.to_path_buf());
    event_loop.run_app(&mut app)?;

    Ok(())
}
