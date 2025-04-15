#![allow(unused)]

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::x11::WindowBuilderExtX11;
use winit::window::WindowBuilder;

mod contree;
mod mempool;
mod renderer;
mod camera;

fn main() {
    // Creating the logger
    env_logger::init();

    // Creating the event loop & window
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut window = WindowBuilder::new().with_title("iVy").build(&event_loop).unwrap();
    let mut renderer = renderer::Renderer::new(&window);

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::KeyboardInput { event, .. } => elwt.exit(),
            WindowEvent::Resized(_) => renderer.resize(&mut window),
            _ => (),
        },
        Event::AboutToWait => renderer.draw(&window),
        _ => (),
    });
}
