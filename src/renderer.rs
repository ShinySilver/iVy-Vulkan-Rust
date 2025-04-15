use ash::ext;
use ash::khr;
use ash::vk;

use std::ffi::CStr;
use log::{debug, info, warn, error};
use winit::window::Window;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct Renderer {
    // Core Vulkan
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family_index: u32,

    // Swapchain
    swapchain_loader: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    framebuffers: Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,

    // Shared Image
    storage_image: vk::Image,
    storage_image_memory: vk::DeviceMemory,
    storage_image_view: vk::ImageView,
    storage_image_format: vk::Format,
    storage_image_extent: (u32, u32),

    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    compute_descriptor_set: vk::DescriptorSet,
    graphics_descriptor_set: vk::DescriptorSet,

    // Pipelines
    compute_pipeline_layout: vk::PipelineLayout,
    compute_pipeline: vk::Pipeline,
    graphics_pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    // Commands
    command_pool: vk::CommandPool,
    compute_command_buffer: vk::CommandBuffer,
    graphics_command_buffers: Vec<vk::CommandBuffer>,

    // Sync
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,

    // Frame
    current_image_index: u32,
    needs_resize: bool,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let entry = ash::Entry::linked();

        // 1. Create instance
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
            .enabled_extension_names(&extensions) // <-- This works fine too!
            .enabled_layer_names(&layers);

        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };

        // 2. Enable validation layer
        let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
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

        // 3. Create surface
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None).unwrap()
        };
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        // 4. Select physical device
        let physical_device = unsafe {
            instance.enumerate_physical_devices().unwrap()
        }[0];

        // 5. Find graphics queue family
        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find(|(_, info)| info.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
                .map(|(index, _)| index as u32)
                .unwrap()
        };

        // 6. Create logical device + queue
        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];

        let device_extensions = [khr::swapchain::NAME.as_ptr()];

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(physical_device, &device_info, None).unwrap() };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // 7. Swapchain loader
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        // 8. Create swapchain-dependent resources
        let mut renderer = Renderer {
            entry,
            instance,
            device,
            surface_loader,
            surface,
            physical_device,
            queue,
            queue_family_index,
            swapchain_loader,
            swapchain: vk::SwapchainKHR::null(),
            swapchain_images: vec![],
            swapchain_image_views: vec![],
            swapchain_format: vk::Format::UNDEFINED,
            swapchain_extent: vk::Extent2D { width: 0, height: 0 },
            framebuffers: vec![],
            render_pass: vk::RenderPass::null(),
            storage_image: vk::Image::null(),
            storage_image_memory: vk::DeviceMemory::null(),
            storage_image_view: vk::ImageView::null(),
            storage_image_format: vk::Format::UNDEFINED,
            storage_image_extent: (0, 0),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            compute_descriptor_set: vk::DescriptorSet::null(),
            graphics_descriptor_set: vk::DescriptorSet::null(),
            compute_pipeline_layout: vk::PipelineLayout::null(),
            compute_pipeline: vk::Pipeline::null(),
            graphics_pipeline_layout: vk::PipelineLayout::null(),
            graphics_pipeline: vk::Pipeline::null(),
            command_pool: vk::CommandPool::null(),
            compute_command_buffer: vk::CommandBuffer::null(),
            graphics_command_buffers: vec![],
            image_available_semaphore: vk::Semaphore::null(),
            render_finished_semaphore: vk::Semaphore::null(),
            in_flight_fence: vk::Fence::null(),
            current_image_index: 0,
            needs_resize: false,
        };

        // 9. Command pool
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(renderer.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        renderer.command_pool = unsafe {
            renderer.device.create_command_pool(&command_pool_info, None).unwrap()
        };

        // 10. Sync objects
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        renderer.image_available_semaphore = unsafe {
            renderer.device.create_semaphore(&semaphore_info, None).unwrap()
        };
        renderer.render_finished_semaphore = unsafe {
            renderer.device.create_semaphore(&semaphore_info, None).unwrap()
        };

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        renderer.in_flight_fence = unsafe {
            renderer.device.create_fence(&fence_info, None).unwrap()
        };

        // 11. Descriptor set layout
        let layout_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT)];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&layout_binding);
        renderer.descriptor_set_layout = unsafe {
            renderer.device.create_descriptor_set_layout(&layout_info, None).unwrap()
        };

        // 12. Full resize setup (also records command buffers)
        renderer.resize(window);

        renderer
    }


    pub fn resize(&mut self, window: &Window) {
        self.needs_resize = false;

        unsafe {
            self.device.device_wait_idle().unwrap();
        }

        // Destroy old resources that depend on window size
        self.cleanup_swapchain_resources();

        // Recreate window-dependent resources
        self.create_swapchain_resources(window);
        self.create_storage_image();
        self.recreate_descriptor_sets();
        self.recreate_command_buffers();
        self.record_pipeline_and_commands();
    }
    pub fn draw(&mut self, window: &Window) {
        if self.needs_resize {
            self.resize(window);
        }
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fence], true, std::u64::MAX)
                .expect("Failed to wait for fence");
            self.device
                .reset_fences(&[self.in_flight_fence])
                .expect("Failed to reset fence");
        }

        let (image_index, _) = unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    std::u64::MAX,
                    self.image_available_semaphore,
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image")
        };
        self.current_image_index = image_index;

        let graphics_cmd = self.graphics_command_buffers[image_index as usize];

        // Submit compute
        let wait_semaphores = &[self.image_available_semaphore];
        let command_buffers = &[self.compute_command_buffer];
        let signal_semaphores = &[self.render_finished_semaphore];
        let compute_submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COMPUTE_SHADER])
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        let submits = &[compute_submit_info];
        unsafe {
            self.device
                .queue_submit(self.queue, submits, self.in_flight_fence)
                .expect("Failed to submit compute command buffer");
        }

        // Submit graphics
        let wait_semaphores = &[self.render_finished_semaphore];
        let graphics_command_buffers = &[graphics_cmd];
        let graphics_submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(graphics_command_buffers);

        let result = unsafe {
            self.device
                .queue_submit(self.queue, &[graphics_submit_info], vk::Fence::null())
        };
        match result {
            Ok(_) => {} // all good
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return,
            Err(vk::Result::SUBOPTIMAL_KHR) => { self.needs_resize = true }
            Err(e) => panic!("Failed to submit draw call: {:?}", e),
        }

        // Present
        let swapchains = &[self.swapchain];
        let image_indices = &[image_index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        };
        match result {
            Ok(_) => {} // all good
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return,
            Err(vk::Result::SUBOPTIMAL_KHR) => { self.needs_resize = true }
            Err(e) => panic!("Failed to present image: {:?}", e),
        }
    }
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

impl Renderer {
    fn create_swapchain_resources(&mut self, window: &Window) {
        let surface_caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .unwrap()
        };

        let extent = match surface_caps.current_extent.width {
            std::u32::MAX => {
                let size = window.inner_size();
                vk::Extent2D {
                    width: size.width,
                    height: size.height,
                }
            }
            _ => surface_caps.current_extent,
        };

        let surface_format = vk::Format::B8G8R8A8_UNORM;
        let present_mode = vk::PresentModeKHR::FIFO;

        let image_count = surface_caps.min_image_count + 1;
        let image_count = if surface_caps.max_image_count > 0 {
            image_count.min(surface_caps.max_image_count)
        } else {
            image_count
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format)
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
                    .format(surface_format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    );
                unsafe { self.device.create_image_view(&view_info, None).unwrap() }
            })
            .collect::<Vec<_>>();

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
                self.device
                    .create_render_pass(&render_pass_info, None)
                    .unwrap()
            }
        };

        let framebuffers = views
            .iter()
            .map(|&view| {
                let attachments = [view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
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
        self.swapchain_format = surface_format;
        self.swapchain_extent = extent;
        self.render_pass = render_pass;
        self.framebuffers = framebuffers;
    }

    fn create_storage_image(&mut self) {
        let extent = self.swapchain_extent;
        let format = vk::Format::R8G8B8A8_UNORM;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
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
            .format(format)
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
        self.storage_image_format = format;
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

    fn create_buffer(
        device: &ash::Device,
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
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };

        let mem_type_index = (0..memory_properties.memory_type_count)
            .find(|&i| {
                let compatible = (mem_req.memory_type_bits & (1 << i)) != 0;
                let supported = memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties);
                compatible && supported
            })
            .expect("No suitable memory type found!");

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };

        unsafe {
            device.bind_buffer_memory(buffer, memory, 0).unwrap();
        }

        (buffer, memory)
    }

    fn recreate_descriptor_sets(&mut self) {
        // Pool
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 2,
        }];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(2);

        let descriptor_pool =
            unsafe { self.device.create_descriptor_pool(&pool_info, None).unwrap() };

        // Allocate
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

        for &set in &descriptor_sets {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&image_info);

            unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        }

        self.descriptor_pool = descriptor_pool;
        self.compute_descriptor_set = descriptor_sets[0];
        self.graphics_descriptor_set = descriptor_sets[1];
    }
    fn recreate_command_buffers(&mut self) {
        let buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(self.swapchain_images.len() as u32 + 1);

        let command_buffers = unsafe {
            self.device.allocate_command_buffers(&buffer_info).unwrap()
        };

        let (graphics_buffers, compute_buffers) = command_buffers.split_at(self.swapchain_images.len());
        self.graphics_command_buffers = graphics_buffers.to_vec();
        self.compute_command_buffer = compute_buffers[0];
    }
    fn record_pipeline_and_commands(&mut self) {
        // Cleanup old pipelines if needed
        unsafe {
            self.device.destroy_pipeline(self.compute_pipeline, None);
            self.device.destroy_pipeline_layout(self.compute_pipeline_layout, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_pipeline_layout(self.graphics_pipeline_layout, None);
        }

        // === CREATE COMPUTE PIPELINE ===
        let primary_ray_shader = load_shader_module!(self.device, "primary_ray.comp");
        let compute_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(primary_ray_shader)
            .name(c"main");

        let descriptor_set_layouts = &[self.descriptor_set_layout];
        let compute_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts);

        self.compute_pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&compute_layout_info, None)
                .unwrap()
        };

        let compute_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(compute_shader_stage_info)
            .layout(self.compute_pipeline_layout);

        self.compute_pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_info], None)
                .unwrap()[0]
        };

        // === CREATE GRAPHICS PIPELINE ===
        let fullscreen_vert_shader = load_shader_module!(self.device, "fullscreen.vert");
        let vert_shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(fullscreen_vert_shader)
            .name(c"main");

        let fullscreen_frag_shader = load_shader_module!(self.device, "fullscreen.frag");
        let frag_shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fullscreen_frag_shader)
            .name(c"main");

        let shader_stages = [vert_shader_stage, frag_shader_stage];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = [vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissor = [vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(self.swapchain_extent)];

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewport)
            .scissors(&scissor);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R | vk::ColorComponentFlags::G |
                    vk::ColorComponentFlags::B | vk::ColorComponentFlags::A
            )];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachment);

        let graphics_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts);

        self.graphics_pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&graphics_layout_info, None)
                .unwrap()
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(self.graphics_pipeline_layout)
            .render_pass(self.render_pass)
            .subpass(0);

        self.graphics_pipeline = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()[0]
        };

        // Destroying the newly created shader module now that the pipeline is built
        unsafe {
            self.device.destroy_shader_module(primary_ray_shader, None);
            self.device.destroy_shader_module(fullscreen_vert_shader, None);
            self.device.destroy_shader_module(fullscreen_frag_shader, None);
        }

        // === RECORD COMMAND BUFFERS ===

        // Record compute command buffer
        unsafe {
            self.device
                .reset_command_buffer(self.compute_command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(self.compute_command_buffer, &begin_info)
                .unwrap();

            self.device.cmd_bind_pipeline(
                self.compute_command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.compute_command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline_layout,
                0,
                &[self.compute_descriptor_set],
                &[],
            );

            // Dispatch work groups
            let (w, h) = self.storage_image_extent;
            self.device.cmd_dispatch(
                self.compute_command_buffer,
                (w + 7) / 8,
                (h + 7) / 8,
                1,
            );

            self.device.end_command_buffer(self.compute_command_buffer).unwrap();
        }

        // Record graphics command buffers (one per framebuffer)
        for (i, &cmd_buf) in self.graphics_command_buffers.iter().enumerate() {
            unsafe {
                self.device
                    .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::empty())
                    .unwrap();

                let begin_info = vk::CommandBufferBeginInfo::default();
                self.device
                    .begin_command_buffer(cmd_buf, &begin_info)
                    .unwrap();

                let clear_value = vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                };
                let clear_values = &[clear_value];
                let render_pass_info = vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[i])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.swapchain_extent,
                    })
                    .clear_values(clear_values);

                self.device.cmd_begin_render_pass(
                    cmd_buf,
                    &render_pass_info,
                    vk::SubpassContents::INLINE,
                );

                self.device.cmd_bind_pipeline(
                    cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipeline,
                );

                self.device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipeline_layout,
                    0,
                    &[self.graphics_descriptor_set],
                    &[],
                );

                self.device.cmd_draw(cmd_buf, 3, 1, 0, 0); // Fullscreen triangle

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

            self.device.destroy_render_pass(self.render_pass, None);

            for view in self.swapchain_image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            self.device.destroy_image_view(self.storage_image_view, None);
            self.device.destroy_image(self.storage_image, None);
            self.device.free_memory(self.storage_image_memory, None);

            self.device.free_command_buffers(
                self.command_pool,
                &[self.compute_command_buffer],
            );
            if self.graphics_command_buffers.len() > 0 {
                self.device.free_command_buffers(
                    self.command_pool,
                    &self.graphics_command_buffers,
                )
            }

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

#[cfg(debug_assertions)]
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
