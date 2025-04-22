#![allow(unused)]

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::event::{Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::x11::WindowBuilderExtX11;
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

mod camera;
mod contree;
mod mempool;
mod renderer;

fn main() {
    // Creating the logger
    env_logger::init();

    // Creating the winit event loop, window and input helper
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut window = WindowBuilder::new()
        .with_title("iVy")
        .build(&event_loop)
        .unwrap();
    let mut input_helper = WinitInputHelper::new();

    // Creating renderer & camera
    let mut renderer = renderer::Renderer::new(&window);
    let mut camera = camera::Camera::new(
        glam::Vec3::new(0.0, 1.0, 5.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let mut projection = camera::Projection::new(45.0_f32.to_radians(), 0.1, 100.0);

    // Main loop
    event_loop.run(move |event, elwt| {
        input_helper.update(&event);
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                        ..
                    }, ..
                } => elwt.exit(),
                WindowEvent::Resized(_) => {
                    renderer.resize(&mut window);
                    projection.update(&input_helper)
                }
                _ => (),
            },
            Event::AboutToWait => {
                camera.update(&input_helper);
                renderer.draw(&window, &projection, &camera)
            }
            _ => (),
        }
    });
}
