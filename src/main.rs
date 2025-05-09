#![allow(unused)]

use fastnoise2::generator::{Generator, prelude::*};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::event::{Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::x11::WindowBuilderExtX11;
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

use std::ops::{Add, Mul};
use std::time::{Duration, Instant};
use glam::UVec3;
use log::info;

mod camera;
mod contree;
mod mempool;
mod renderer;
mod world;
mod worldgen;

use crate::contree::Contree;

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

    // Creating a heightmap for the world
    const NODE_WIDTH: u32 = 4;
    const WORLD_DEPTH: u32 = 6;
    const WORLD_WIDTH: u32 = NODE_WIDTH.pow(WORLD_DEPTH);
    let mut world_heightmap = vec![0.0; WORLD_WIDTH.pow(2) as usize];
    let node = opensimplex2()
        .fbm(0.65, 0.5, 4, 2.5)
        .domain_scale(0.066)
        .domain_warp_gradient(0.2, 2.0)
        .domain_warp_progressive(0.7, 0.5, 2, 2.5)
        .mul(100.0)
        .add(50.0)
        .build();
    let start = Instant::now();
    let min_max = node.gen_uniform_grid_2d(
        &mut world_heightmap,
        0,
        0,
        WORLD_WIDTH as i32,
        WORLD_WIDTH as i32,
        0.02,
        1337,
    );
    let elapsed = start.elapsed();
    info!("Generated the {WORLD_WIDTH}x{WORLD_WIDTH} heightmap in {:?}", elapsed);

    // Creating a 64-tree for the world
    let start = Instant::now();
    let mut tree = Contree::new(WORLD_DEPTH as usize);
    for x in 0..WORLD_WIDTH {
        for y in 0..WORLD_WIDTH {
            let z = world_heightmap[(x + y * WORLD_WIDTH) as usize] as u32;
            tree.set(UVec3 { x, y, z }, 1u8);
        }
    }
    let elapsed = start.elapsed();
    info!("Generated the {WORLD_WIDTH}x{WORLD_WIDTH}x{WORLD_WIDTH} 64-tree in {:?}", elapsed);

    // Creating renderer & camera
    let mut renderer = renderer::Renderer::new(&window, tree.nodes.raw());
    let mut camera = camera::Camera::new(
        glam::Vec3::new(0.0, 1.0, 5.0),
        glam::Vec3::X,
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
                    info!("Average frame time: {}ms ({} FPS)", duration.as_millis(), frame_count as f32 / duration.as_secs_f32());
                    start = Instant::now();
                    frame_count = 0;
                }
            }
            _ => (),
        }
    });
}
