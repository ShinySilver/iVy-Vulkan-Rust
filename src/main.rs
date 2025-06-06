#![allow(unused)]

use fastnoise2::generator::{prelude::*, Generator};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::event::{Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::x11::WindowBuilderExtX11;
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

use std::ops::{Add, Mul};
use std::time::{Duration, Instant};
use glam::{uvec3, vec3, UVec3};
use log::{error, info};
use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::utils::sparse_bitmask::SparseBitmask;

mod camera;
mod renderer;
mod utils;
pub mod world;

use crate::utils::sparse_tree::*;
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
    let world = World::new(5, 145904);
    info!("World has a memory footprint of {} MB ({} MB for nodes, {} MB for leaves)",
        (world.data.nodes.size()*96 + world.data.voxels .size()*16) as f64 / 8. / 1e6,
        (world.data.nodes.size()*96) as f64 / 8. / 1e6, (world.data.voxels.size()*16) as f64 / 8. / 1e6);

    // Creating renderer & camera
    let mut renderer = renderer::Renderer::new(&window, world.raw_node_data(), world.raw_voxel_data());
    let mut camera = camera::Camera::new(
        vec3(0.0, 800.0, 1024.0),
        vec3(1.0, -0.45, -0.8),
        glam::Vec3::Y,
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
                        projection.update(&input_helper)
                    }
                    _ => (),
                }
            }
            Event::AboutToWait => {
                camera.update(&window, &input_helper);
                renderer.draw(&window, &projection, &camera);
                frame_count += 1;
                if frame_count % 1000 == 0 {
                    let duration = start.elapsed();
                    info!("Average frame time: {:.2}ms ({:.2} FPS)", duration.as_millis() as f32 / 1000., frame_count as f32 / duration.as_secs_f32());
                    start = Instant::now();
                    frame_count = 0;
                }
            }
            _ => (),
        }
    });
}
