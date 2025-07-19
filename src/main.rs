use winit::event::{Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

use glam::vec3;
use log::info;
use std::time::Instant;

mod camera;
mod renderer;
mod utils;
mod world;
mod voxels;

use crate::world::World;

fn main() {
    // Creating the logger
    env_logger::init();

    // Creating the winit event loop, window, gui context and input helper
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut window = WindowBuilder::new()
        .with_title("iVy")
        .build(&event_loop)
        .unwrap();
    let mut input_helper = WinitInputHelper::new();

    // Generating world
    let world_depth = 5u32;
    let world_width = 4f32.powf(world_depth as f32);
    let mut world = World::new(world_depth, 145904);
    info!("World has a memory footprint of {} MB ({} MB for nodes, {} MB for leaves)",
        (world.data.nodes.size()*96 + world.data.voxels .size()*16) as f64 / 8. / 1e6,
        (world.data.nodes.size()*96) as f64 / 8. / 1e6, (world.data.voxels.size()*16) as f64 / 8. / 1e6);

    // Creating renderer & camera
    let mut renderer = renderer::Renderer::new(&window, world.raw_node_data(), world.raw_voxel_data());
    let mut player = camera::Camera::new(
        vec3(0.0, 0.8 * world_width, 0.5 * world_width),
        vec3(1.0, -0.45, -0.5),
    );
    let mut projection = camera::Projection::new(45.0_f32.to_radians(), 0.1, 100.0);

    // Main loop
    let mut frame_count = 0u32;
    let mut start = Instant::now();
    event_loop.run(move |event, elwt| {
        input_helper.update(&event);
        match event {
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                            ..
                        }, ..
                    } => elwt.exit(),
                    WindowEvent::Resized(_) => {
                        renderer.resize(&mut window);
                        projection.update(&input_helper);
                        start = Instant::now();
                        frame_count = 0;
                    }
                    _ => (),
                }
            }
            Event::AboutToWait => {
                world.update();
                player.update(&window, &input_helper);
                renderer.draw(&window, &projection, &player, &world);
                frame_count += 1;
                if frame_count % 8000 == 0 {
                    let duration = start.elapsed();
                    info!("Average frame time: {:.2}ms ({:.2} FPS)", duration.as_millis() as f32 / frame_count as f32, frame_count as f32 / duration.as_secs_f32());
                    start = Instant::now();
                    frame_count = 0;
                }
            }
            _ => (),
        }
    }).unwrap();
}
