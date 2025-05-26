use ash::ext;
use ash::khr;
use ash::vk;

use std::ffi::CStr;
use log::{debug, info, warn, error};
use winit::window::Window;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::camera::{Camera, Projection};
use crate::utils::sparse_tree::Node;

pub struct Renderer {
    // ----------------------------
    // Vulkan Core Handles
    // ----------------------------
    /// Vulkan entry point (loaded dynamically)
    entry: ash::Entry,

    /// Vulkan instance (represents the Vulkan context)
    instance: ash::Instance,

    /// Logical device (interface to the GPU)
    device: ash::Device,

    /// Selected physical GPU
    physical_device: vk::PhysicalDevice,

    /// Queue handle to submit commands
    queue: vk::Queue,

    /// Index of the queue family that supports our needs (graphics/compute)
    queue_family_index: u32,

    // ----------------------------
    // Surface (Window Integration)
    // ----------------------------
    /// Surface extension loader (interface between Vulkan and window system)
    surface_loader: khr::surface::Instance,

    /// The actual surface (window connection)
    surface: vk::SurfaceKHR,

    /// Format for displaying images to the surface (e.g., sRGB)
    surface_format: vk::Format,

    // ----------------------------
    // Swapchain (Backbuffer Images)
    // ----------------------------
    /// Swapchain loader (interface to manage swapchain)
    swapchain_loader: khr::swapchain::Device,

    /// Swapchain handle (set of images used for rendering/displaying)
    swapchain: vk::SwapchainKHR,

    /// Images in the swapchain
    swapchain_images: Vec<vk::Image>,

    /// Views of the swapchain images (like textures)
    swapchain_image_views: Vec<vk::ImageView>,

    /// Resolution of the swapchain images
    swapchain_extent: vk::Extent2D,

    /// Framebuffers for rendering to swapchain images
    framebuffers: Vec<vk::Framebuffer>,

    /// Render pass used for graphics rendering
    render_pass: vk::RenderPass,

    // ----------------------------
    // Storage Image (Compute Output Target)
    // ----------------------------
    /// Image that compute shader writes to
    storage_image: vk::Image,

    /// Memory bound to the storage image
    storage_image_memory: vk::DeviceMemory,

    /// View into the storage image (for shaders to access)
    storage_image_view: vk::ImageView,

    /// Size of the storage image
    storage_image_extent: (u32, u32),

    // ----------------------------
    // Camera Uniform Buffer (UBO)
    // ----------------------------
    /// Uniform buffer for camera data (view/projection matrices)
    camera_buffer: vk::Buffer,

    /// GPU memory for the camera buffer
    camera_buffer_memory: vk::DeviceMemory,

    /// CPU-side pointer to mapped camera buffer (for updates)
    camera_mapped_ptr: *mut CameraUBO,

    // ----------------------------
    // Scene Data: 64-tree Buffer
    // ----------------------------
    /// GPU buffer for node data
    node_buffer: vk::Buffer,

    /// Memory for the node buffer
    node_buffer_memory: vk::DeviceMemory,

    /// Number of nodes in the tree
    node_count: usize,

    // ----------------------------
    // Descriptor Sets & Layouts
    // ----------------------------
    /// Pool for allocating descriptor sets (GPU resource bindings)
    descriptor_pool: vk::DescriptorPool,

    /// Layout describing bindings for compute shaders
    descriptor_set_layout: vk::DescriptorSetLayout,

    /// Descriptor set used by compute shaders
    compute_descriptor_set: vk::DescriptorSet,

    /// Descriptor set used by graphics shaders (optional)
    graphics_descriptor_set: vk::DescriptorSet,

    // ----------------------------
    // Pipelines (Compute & Graphics)
    // ----------------------------
    /// Pipeline layout for compute shaders (descriptor bindings)
    compute_pipeline_layout: vk::PipelineLayout,

    /// Compute pipeline (e.g., raytracing)
    compute_pipeline: vk::Pipeline,

    /// Pipeline layout for graphics shaders
    graphics_pipeline_layout: vk::PipelineLayout,

    /// Graphics pipeline (e.g., drawing a textured quad)
    graphics_pipeline: vk::Pipeline,

    // ----------------------------
    // Command Buffers
    // ----------------------------
    /// Command pool for allocating command buffers
    command_pool: vk::CommandPool,

    /// Command buffers for drawing frames
    command_buffers: Vec<vk::CommandBuffer>,

    // ----------------------------
    // Synchronization Objects
    // ----------------------------
    /// Semaphore to signal when a swapchain image is ready
    image_available_semaphore: vk::Semaphore,

    /// Fence to signal when rendering is complete
    in_flight_fence: vk::Fence,

    // ----------------------------
    // Frame State
    // ----------------------------
    /// Index of the currently rendered image
    current_image_index: u32,

    /// Number of images in the swapchain
    image_count: u32,

    /// Flag to indicate the swapchain needs resizing
    needs_resize: bool,
}


macro_rules! load_shader_module {
    ($device:expr, $path:expr) => {{
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/", $path, ".spv"));
        assert!(bytes.len() % 4 == 0, "SPIR-V file is not aligned to u32");

        let shader_code: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: bytes.len(),
            p_code: shader_code.as_ptr(),
            ..Default::default()
        };

        unsafe {
            $device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create shader module")
        }
    }};
}

#[repr(C)]
pub struct CameraUBO {
    pub position: glam::Vec4,
    pub inv_view: glam::Mat4,
    pub inv_proj: glam::Mat4,
}

impl Renderer {
    pub fn new(window: &Window, nodes: &Vec<Node>) -> Self {
        //
        // ─────────────────────────────────────────────────────────────
        // 1. Load Vulkan Entry Point
        // ─────────────────────────────────────────────────────────────
        // Vulkan uses dynamic loading. The Entry object is our access to Vulkan functions.
        //
        let entry = ash::Entry::linked();

        //
        // ─────────────────────────────────────────────────────────────
        // 2. Create Vulkan Instance
        // ─────────────────────────────────────────────────────────────
        // The instance is the connection between our application and the Vulkan driver.
        // We provide basic application info, extensions required for windowing,
        // and enable validation layers for debugging.
        //
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"iVY")
            .application_version(0)
            .engine_name(c"NoEngine")
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let window_handle = window.window_handle().unwrap().as_raw();
        let display_handle = window.display_handle().unwrap().as_raw();

        let mut extensions = ash_window::enumerate_required_extensions(display_handle)
            .unwrap()
            .to_vec();
        extensions.push(ext::debug_utils::NAME.as_ptr());

        let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);

        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };

        //
        // ─────────────────────────────────────────────────────────────
        // 3. Enable Debugging via Validation Layers (optional but helpful)
        // ─────────────────────────────────────────────────────────────
        // This enables messages from Vulkan that help track incorrect usage.
        //
        let debug_utils = ext::debug_utils::Instance::new(&entry, &instance);
        let debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT {
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(vulkan_debug_utils_callback),
            ..Default::default()
        };
        let utils_messenger = unsafe {
            debug_utils
                .create_debug_utils_messenger(&debug_create_info, None)
                .expect("Error creating utils messenger")
        };

        //
        // ─────────────────────────────────────────────────────────────
        // 4. Create Window Surface
        // ─────────────────────────────────────────────────────────────
        // This allows Vulkan to present rendered images to a platform-specific window.
        //
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                .unwrap()
        };
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        //
        // ─────────────────────────────────────────────────────────────
        // 5. Choose a Physical Device (GPU)
        // ─────────────────────────────────────────────────────────────
        // Select the first available physical device.
        // Normally, you'd pick based on features and performance.
        //
        let physical_device = unsafe { instance.enumerate_physical_devices().unwrap() }[0];
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        //
        // ─────────────────────────────────────────────────────────────
        // 6. Query Surface Capabilities
        // ─────────────────────────────────────────────────────────────
        // Determine the supported properties of the swapchain images (e.g., count, size).
        //
        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        let mut image_count = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0 {
            image_count = image_count.min(surface_caps.max_image_count);
        }

        //
        // ─────────────────────────────────────────────────────────────
        // 7. Select Queue Family (Graphics & Compute Support)
        // ─────────────────────────────────────────────────────────────
        // Vulkan requires you to choose which queue(s) to use for operations.
        //
        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find(|(_, info)| {
                    info.queue_flags
                        .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .map(|(index, _)| index as u32)
                .unwrap()
        };

        //
        // ─────────────────────────────────────────────────────────────
        // 8. Create Logical Device and Retrieve Queue Handle
        // ─────────────────────────────────────────────────────────────
        // The logical device provides access to Vulkan functions and queues.
        //
        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let device_extensions = [khr::swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions);
        let device =
            unsafe { instance.create_device(physical_device, &device_info, None).unwrap() };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        //
        // ─────────────────────────────────────────────────────────────
        // 9. Swapchain & Command Infrastructure
        // ─────────────────────────────────────────────────────────────
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool =
            unsafe { device.create_command_pool(&command_pool_info, None).unwrap() };

        let buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(image_count);

        let command_buffers = unsafe { device.allocate_command_buffers(&buffer_info).unwrap() };

        //
        // ─────────────────────────────────────────────────────────────
        // 10. Synchronization Primitives
        // ─────────────────────────────────────────────────────────────
        // Used for GPU-CPU and inter-GPU coordination.
        //
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let image_available_semaphore =
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight_fence = unsafe { device.create_fence(&fence_info, None).unwrap() };

        //
        // ─────────────────────────────────────────────────────────────
        // 11. Camera Uniform Buffer (CPU-visible)
        // ─────────────────────────────────────────────────────────────
        // Uniform buffers are typically updated every frame.
        //
        let camera_ubo_size = size_of::<CameraUBO>() as vk::DeviceSize;
        let camera_buffer = {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(camera_ubo_size)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            unsafe { device.create_buffer(&buffer_info, None).unwrap() }
        };

        let camera_buffer_memory = {
            let mem_requirements = unsafe { device.get_buffer_memory_requirements(camera_buffer) };
            let memory_type_index = memory_properties
                .memory_types
                .iter()
                .enumerate()
                .find(|(i, mem_type)| {
                    (mem_requirements.memory_type_bits & (1u32 << i)) != 0
                        && mem_type.property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                })
                .map(|(i, _)| i as u32)
                .expect("No suitable memory type");

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(memory_type_index);

            unsafe { device.allocate_memory(&alloc_info, None).unwrap() }
        };

        unsafe {
            device
                .bind_buffer_memory(camera_buffer, camera_buffer_memory, 0)
                .unwrap();
        }

        let camera_mapped_ptr = unsafe {
            device
                .map_memory(
                    camera_buffer_memory,
                    0,
                    camera_ubo_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut CameraUBO
        };

        //
        // ─────────────────────────────────────────────────────────────
        // 12. Node Buffer Transfer: Staging to Device Local
        // ─────────────────────────────────────────────────────────────
        // GPU-visible (but CPU-inaccessible) memory, using a staging buffer for upload.
        //
        let buffer_size = (nodes.len() * size_of::<Node>()) as vk::DeviceSize;
        let (staging_buffer, staging_memory) = create_buffer(
            &device,
            &memory_properties,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(staging_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .unwrap() as *mut Node;
            data_ptr.copy_from_nonoverlapping(nodes.as_ptr(), nodes.len());
            device.unmap_memory(staging_memory);
        }

        let (node_buffer, node_buffer_memory) = create_buffer(
            &device,
            &memory_properties,
            buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        copy_buffer(
            &device,
            command_pool,
            queue,
            staging_buffer,
            node_buffer,
            buffer_size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        }

        //
        // ─────────────────────────────────────────────────────────────
        // 13. Descriptor Set Layout
        // ─────────────────────────────────────────────────────────────
        // Describes how shaders can access buffers/images.
        //
        let layout_binding = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_binding);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None).unwrap() };

        //
        // ─────────────────────────────────────────────────────────────
        // 14. Load Shader Modules (Compute, Vertex, Fragment)
        // ─────────────────────────────────────────────────────────────
        // Load SPIR-V shader binaries and wrap them in Vulkan shader modules.
        // These are then plugged into the pipeline stage descriptions.
        //
        let primary_ray_shader = load_shader_module!(device, "primary_ray.comp");
        let compute_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(primary_ray_shader)
            .name(c"main");

        let fullscreen_vert_shader = load_shader_module!(device, "fullscreen.vert");
        let vert_shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(fullscreen_vert_shader)
            .name(c"main");

        let fullscreen_frag_shader = load_shader_module!(device, "fullscreen.frag");
        let frag_shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fullscreen_frag_shader)
            .name(c"main");


        //
        // ─────────────────────────────────────────────────────────────
        // 15. Creating the Compute Pipeline
        // ─────────────────────────────────────────────────────────────
        // A compute pipeline consists of a single shader stage and a pipeline layout.
        // The layout describes the descriptor sets used by the shader (e.g., buffers, images).
        //
        let descriptor_set_layouts = &[descriptor_set_layout];

        let compute_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts);

        let compute_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&compute_layout_info, None)
                .unwrap()
        };

        let compute_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(compute_shader_stage_info)
            .layout(compute_pipeline_layout);

        let compute_pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_info], None)
                .unwrap()[0]
        };

        //
        // ─────────────────────────────────────────────────────────────
        // 16. Creating the Graphics Pipeline
        // ─────────────────────────────────────────────────────────────
        // The graphics pipeline uses a vertex and fragment shader, and a lot more state.
        // It includes viewport/scissor info, rasterization, blending, and a render pass.
        //

        let shader_stages = [vert_shader_stage, frag_shader_stage];

        // We're drawing a fullscreen quad with no vertex input, so we use defaults.
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        // Use triangle list topology for drawing
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        // Viewport and scissor will be set dynamically per-frame
        let dynamic_states = vec![
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
        ];

        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        // Basic rasterization setup — fill mode, no culling, clockwise front face
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);

        // No multisampling
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Enable color output with all color components (RGBA), no blending
        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R |
                    vk::ColorComponentFlags::G |
                    vk::ColorComponentFlags::B |
                    vk::ColorComponentFlags::A
            )];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachment);

        // Layout used by the graphics pipeline (same descriptor layout as compute)
        let graphics_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts);

        let graphics_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&graphics_layout_info, None)
                .unwrap()
        };

        // The render pass describes the framebuffer attachments (color, depth, etc.)
        let surface_format = vk::Format::B8G8R8A8_UNORM;

        let render_pass = {
            let color_attachment = [vk::AttachmentDescription::default()
                .format(surface_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

            let color_attachment_ref = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];

            let subpass = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_attachment_ref)];

            let render_pass_info = vk::RenderPassCreateInfo::default()
                .attachments(&color_attachment)
                .subpasses(&subpass);

            unsafe {
                device
                    .create_render_pass(&render_pass_info, None)
                    .unwrap()
            }
        };

        // Tell Vulkan that viewport and scissor will be dynamic
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Final graphics pipeline creation
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .viewport_state(&viewport_state_info)
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .dynamic_state(&dynamic_state_info)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(graphics_pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()[0]
        };

        //
        // ─────────────────────────────────────────────────────────────
        // 17. Destroy Shader Modules After Pipeline Creation
        // ─────────────────────────────────────────────────────────────
        // The compiled SPIR-V shaders are loaded into Vulkan modules, but we don’t
        // need them anymore after pipeline creation.
        //
        unsafe {
            device.destroy_shader_module(primary_ray_shader, None);
            device.destroy_shader_module(fullscreen_vert_shader, None);
            device.destroy_shader_module(fullscreen_frag_shader, None);
        };

        //
        // ─────────────────────────────────────────────────────────────
        // 18. Construct Final Renderer Struct
        // ─────────────────────────────────────────────────────────────
        // At this point, all Vulkan resources have been created. We now wrap everything
        // up into the main `Renderer` struct to encapsulate and manage the Vulkan state.
        //
        let mut renderer = Renderer {
            entry,
            instance,
            device,
            surface_loader,
            surface,
            surface_format,
            physical_device,
            queue,
            queue_family_index,
            swapchain_loader,
            swapchain: vk::SwapchainKHR::null(),
            swapchain_images: vec![],
            swapchain_image_views: vec![],
            swapchain_extent: vk::Extent2D { width: 0, height: 0 },
            framebuffers: vec![],
            render_pass,
            storage_image: vk::Image::null(),
            storage_image_memory: vk::DeviceMemory::null(),
            storage_image_view: vk::ImageView::null(),
            storage_image_extent: (0, 0),
            camera_buffer,
            camera_buffer_memory,
            camera_mapped_ptr,
            node_buffer,
            node_buffer_memory,
            node_count: nodes.len(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set_layout,
            compute_descriptor_set: vk::DescriptorSet::null(),
            graphics_descriptor_set: vk::DescriptorSet::null(),
            compute_pipeline_layout,
            compute_pipeline,
            graphics_pipeline_layout,
            graphics_pipeline,
            command_pool,
            command_buffers: command_buffers.to_vec(),
            image_available_semaphore,
            in_flight_fence,
            current_image_index: 0,
            image_count,
            needs_resize: false,
        };

        //
        // ─────────────────────────────────────────────────────────────
        // 19. Perform Initial Setup of Swapchain and Framebuffers
        // ─────────────────────────────────────────────────────────────
        // This call ensures that the swapchain images, views, and framebuffers
        // are created based on the current window size.
        //
        renderer.resize(window);

        renderer
    }

    pub fn resize(&mut self, window: &Window) {
        self.needs_resize = false;

        unsafe {
            self.device.device_wait_idle().unwrap();
        }

        // First of all, clean up all the extent-dependant resources
        self.cleanup_swapchain_resources();

        // Then, recreate them
        self.recreate_swapchain(window);
        self.recreate_storage_image();
        self.recreate_descriptor_sets();

        // And at last, reset & record the command buffer again, with the correct compute shader dispatch size
        self.record_commands();
    }

    pub fn draw(&mut self, window: &Window, projection: &Projection, camera: &Camera) {
        // If during last frame the swapchain was marked as suboptimal, we ensure a resize here
        if self.needs_resize {
            self.resize(window);
        }

        // Sync with the GPU for the oldest in-flight fence
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
                .expect("Failed to wait for fence");
            self.device
                .reset_fences(&[self.in_flight_fence])
                .expect("Failed to reset fence");
        }

        // Acquire next swapchain image
        let (image_index, _) = unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available_semaphore,
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image")
        };
        self.current_image_index = image_index;

        // Update the camera UBO
        let inv_view = camera.view_matrix().inverse();
        let inv_proj = projection.projection_matrix().inverse();
        let ubo_data = CameraUBO {
            position: camera.position.extend(1.0),
            inv_view,
            inv_proj,
        };
        unsafe { std::ptr::copy_nonoverlapping(&ubo_data, self.camera_mapped_ptr, 1) };

        // Submit the commands for both compute and graphics
        let wait_semaphores = [self.image_available_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let cmd_buf = [self.command_buffers[image_index as usize]];
        let submit_info = [vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmd_buf)
            .signal_semaphores(&[])
        ];
        let result = unsafe {
            self.device
                .queue_submit(self.queue, &submit_info, self.in_flight_fence)
        };

        // Handle out of date and suboptimal swapchain
        match result {
            Ok(_) => {} // all good
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return,
            Err(vk::Result::SUBOPTIMAL_KHR) => { self.needs_resize = true }
            Err(e) => panic!("Failed to submit draw call: {:?}", e),
        }

        // Present framebuffer
        let swapchains = &[self.swapchain];
        let image_indices = &[image_index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(swapchains)
            .image_indices(image_indices);
        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        };

        // Handle out of date and suboptimal swapchain again
        match result {
            Ok(_) => {} // all good
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return,
            Err(vk::Result::SUBOPTIMAL_KHR) => { self.needs_resize = true }
            Err(e) => panic!("Failed to present image: {:?}", e),
        }
    }
}

impl Renderer {
    fn recreate_swapchain(&mut self, window: &Window) {
        let surface_caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .unwrap()
        };
        let extent = match surface_caps.current_extent.width {
            u32::MAX => {
                let size = window.inner_size();
                vk::Extent2D {
                    width: size.width,
                    height: size.height,
                }
            }
            _ => surface_caps.current_extent,
        };
        let present_mode = vk::PresentModeKHR::FIFO;
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(self.image_count)
            .image_format(self.surface_format)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);
        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap()
        };
        let images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(swapchain)
                .unwrap()
        };
        let views = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.surface_format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    );
                unsafe { self.device.create_image_view(&view_info, None).unwrap() }
            })
            .collect::<Vec<_>>();
        let framebuffers = views
            .iter()
            .map(|&view| {
                let attachments = [view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(self.render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                unsafe { self.device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect();
        self.swapchain = swapchain;
        self.swapchain_images = images;
        self.swapchain_image_views = views;
        self.swapchain_extent = extent;
        self.framebuffers = framebuffers;
    }

    fn recreate_storage_image(&mut self) {
        let extent = self.swapchain_extent;
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(self.surface_format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { self.device.create_image(&image_info, None).unwrap() };

        let mem_requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let mut mem_type_index = u32::MAX;
        let mem_properties = unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        };
        for (i, mem_type) in mem_properties.memory_types.iter().enumerate() {
            if (mem_requirements.memory_type_bits & (1 << i)) != 0 && mem_type.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
                mem_type_index = i as u32;
                break;
            }
        }
        if (mem_type_index == u32::MAX) { error!("Failed to find suitable memory type!") }

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe { self.device.allocate_memory(&alloc_info, None).unwrap() };

        unsafe {
            self.device.bind_image_memory(image, memory, 0).unwrap();
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(self.surface_format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        let image_view = unsafe { self.device.create_image_view(&view_info, None).unwrap() };

        self.storage_image = image;
        self.storage_image_memory = memory;
        self.storage_image_view = image_view;
        self.storage_image_extent = (extent.width, extent.height);

        // === IMAGE FORMATING ===

        // Now, we are creating a one-time command to ensure the image is in the right format for our re-usable command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);

        let command_buffer = [unsafe {
            self.device.allocate_command_buffers(&alloc_info).unwrap()[0]
        }];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.begin_command_buffer(command_buffer[0], &begin_info).unwrap();
        }

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }
            );

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer[0],
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        unsafe {
            self.device.end_command_buffer(command_buffer[0]).unwrap();

            let submit_info = [vk::SubmitInfo::default()
                .command_buffers(&command_buffer)];

            self.device.queue_submit(self.queue, &submit_info, vk::Fence::null()).unwrap();
            self.device.queue_wait_idle(self.queue).unwrap();
            self.device.free_command_buffers(self.command_pool, &command_buffer);
        }
    }

    fn recreate_descriptor_sets(&mut self) {
        // Recreating the descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(3);
        let descriptor_pool =
            unsafe { self.device.create_descriptor_pool(&pool_info, None).unwrap() };

        // Allocating descriptor sets
        let layouts = [self.descriptor_set_layout; 2];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets =
            unsafe { self.device.allocate_descriptor_sets(&alloc_info).unwrap() };
        let image_info = [vk::DescriptorImageInfo {
            image_view: self.storage_image_view,
            image_layout: vk::ImageLayout::GENERAL,
            sampler: vk::Sampler::null(),
        }];
        let camera_buffer_info = vk::DescriptorBufferInfo {
            buffer: self.camera_buffer,
            offset: 0,
            range: size_of::<CameraUBO>() as vk::DeviceSize,
        };
        let node_buffer_info = vk::DescriptorBufferInfo {
            buffer: self.node_buffer,
            offset: 0,
            range: (self.node_count * size_of::<Node>()) as vk::DeviceSize,
        };
        for &set in &descriptor_sets {
            let write = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&camera_buffer_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&node_buffer_info)),
            ];
            unsafe { self.device.update_descriptor_sets(&write, &[]) };
        }
        self.descriptor_pool = descriptor_pool;
        self.compute_descriptor_set = descriptor_sets[0];
        self.graphics_descriptor_set = descriptor_sets[1];
    }

    fn record_commands(&mut self) {
        for (i, &cmd_buf) in self.command_buffers.iter().enumerate() {
            unsafe {
                self.device
                    .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::empty())
                    .unwrap();

                let begin_info = vk::CommandBufferBeginInfo::default();
                self.device.begin_command_buffer(cmd_buf, &begin_info).unwrap();

                // Compute pass
                self.device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.compute_pipeline_layout,
                    0,
                    &[self.compute_descriptor_set],
                    &[],
                );

                let (w, h) = self.storage_image_extent;
                self.device.cmd_dispatch(cmd_buf, (w + 7) / 8, (h + 7) / 8, 1);

                // Barrier between compute and graphics
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.storage_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

                self.device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );

                // Graphics pass
                let clear_value = vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                };
                let binding = [clear_value];
                let render_pass_info = vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[i])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.swapchain_extent,
                    })
                    .clear_values(&binding);
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.swapchain_extent.width as f32,
                    height: self.swapchain_extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain_extent,
                };
                self.device.cmd_set_viewport(cmd_buf, 0, &[viewport]);
                self.device.cmd_set_scissor(cmd_buf, 0, &[scissor]);
                self.device.cmd_begin_render_pass(
                    cmd_buf,
                    &render_pass_info,
                    vk::SubpassContents::INLINE,
                );
                self.device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, self.graphics_pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipeline_layout,
                    0,
                    &[self.graphics_descriptor_set],
                    &[],
                );

                self.device.cmd_draw(cmd_buf, 3, 1, 0, 0);

                self.device.cmd_end_render_pass(cmd_buf);
                self.device.end_command_buffer(cmd_buf).unwrap();
            }
        }
    }

    fn cleanup_swapchain_resources(&mut self) {
        unsafe {
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer, None);
            }

            for view in self.swapchain_image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            self.device.destroy_image_view(self.storage_image_view, None);
            self.device.destroy_image(self.storage_image, None);
            self.device.free_memory(self.storage_image_memory, None);

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    let ty = format!("{:?}", message_type).to_lowercase();
    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            debug!(target: "iVy::vulkan", "{:?}", message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!(target: "iVy::vulkan", "{:?}", message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!(target: "iVy::vulkan", "{:?}", message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!(target: "iVy::vulkan", "{:?}", message)
        }
        _ => debug!("[Vulkan][unknown] {:?}", message),
    }
    vk::FALSE
}

fn create_buffer(device: &ash::Device,
                 memory_properties: &vk::PhysicalDeviceMemoryProperties,
                 size: vk::DeviceSize,
                 usage: vk::BufferUsageFlags,
                 properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None).unwrap() };
    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_type_index = memory_properties.memory_types.iter().enumerate()
        .find(|(i, mem_type)| {
            (mem_requirements.memory_type_bits & (1u32 << i)) != 0 &&
                mem_type.property_flags.contains(properties)
        })
        .map(|(i, _)| i as u32)
        .expect("Failed to find suitable memory type");

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let buffer_memory = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };

    unsafe {
        device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();
    }

    (buffer, buffer_memory)
}

fn copy_buffer(device: &ash::Device,
               command_pool: vk::CommandPool,
               graphics_queue: vk::Queue,
               src: vk::Buffer,
               dst: vk::Buffer,
               size: vk::DeviceSize,
) {
    let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe {
        device.allocate_command_buffers(&command_buffer_alloc_info)
            .unwrap()[0]
    };

    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device.begin_command_buffer(command_buffer, &begin_info).unwrap();
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        device.cmd_copy_buffer(command_buffer, src, dst, &[copy_region]);
        device.end_command_buffer(command_buffer).unwrap();

        let binding = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&binding);

        device.queue_submit(graphics_queue, &[submit_info], vk::Fence::null()).unwrap();
        device.queue_wait_idle(graphics_queue).unwrap();
        device.free_command_buffers(command_pool, &[command_buffer]);
    }
}
