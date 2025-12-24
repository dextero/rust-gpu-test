use anyhow::Result;
use std::{borrow::Cow, rc::Rc};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, BufferDescriptor, BufferUsages,
    CommandEncoder, ComputePipeline, Device, MapMode, Queue, ShaderModuleDescriptor, ShaderSource,
    Texture, TextureFormat, TextureUsages, util::DeviceExt, wgt::TextureDescriptor,
};

struct GpuFuncInvoker {
    device: Rc<Device>,
    pipeline: ComputePipeline,
}

impl GpuFuncInvoker {
    fn from_source(device: Rc<Device>, label: &str, shader_source: Cow<'static, str>) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", label)),
            source: ShaderSource::Wgsl(shader_source),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", label)),
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });

        Self { device, pipeline }
    }

    fn invoke<'a>(
        &self,
        encoder: &mut CommandEncoder,
        args: impl IntoIterator<Item = BindingResource<'a>>,
        workgroup_size: (u32, u32),
    ) {
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("calc_sizes_bind_group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &args
                .into_iter()
                .enumerate()
                .map(|(idx, resource)| BindGroupEntry {
                    binding: idx.try_into().unwrap(),
                    resource,
                })
                .collect::<Vec<_>>(),
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("calc_sizes"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroup_size.0, workgroup_size.1, 1);
    }
}

macro_rules! static_func_invoker {
    ($device:expr, $src:literal) => {{
        GpuFuncInvoker::from_source(
            $device,
            $src,
            std::borrow::Cow::Borrowed(include_str!($src)),
        )
    }};
}

struct GpuAnsiEncoder {
    device: Rc<Device>,
    queue: Rc<Queue>,
    calc_sizes_invoker: GpuFuncInvoker,
    prefix_sum_invoker: GpuFuncInvoker,
    encode_invoker: GpuFuncInvoker,
}

impl GpuAnsiEncoder {
    pub async fn new(device: Rc<Device>, queue: Rc<Queue>) -> Result<Self> {
        let calc_sizes_invoker = static_func_invoker!(device.clone(), "wgsl/calc_sizes.wgsl");
        let prefix_sum_invoker = static_func_invoker!(device.clone(), "wgsl/prefix_sum.wgsl");
        let encode_invoker = static_func_invoker!(device.clone(), "wgsl/encode_ansi.wgsl");

        Ok(Self {
            device,
            queue,
            calc_sizes_invoker,
            prefix_sum_invoker,
            encode_invoker,
        })
    }

    pub async fn ansi_from_texture(&self, texture: &Texture) -> Result<String> {
        let texture_size = texture.size();
        let num_pixels = texture_size.width * texture_size.height;
        let offsets_size_bytes = usize::try_from(num_pixels)? * std::mem::size_of::<u32>();
        let offsets_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("sizes_offsets"),
            size: offsets_size_bytes.try_into()?,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let total_size_device_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("total_size_device"),
            size: std::mem::size_of::<u32>().try_into().unwrap(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let total_size_host_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("total_size_host"),
            size: std::mem::size_of::<u32>().try_into().unwrap(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let max_output_size_bytes: usize = usize::try_from(num_pixels)?
            * "\x1b[38;2;255;255;255m\x1b[48;2;255;255;255m\u{2580}".len();
        let rounded_max_output_size_bytes = max_output_size_bytes.div_ceil(4) * 4;
        let output_device_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("ansi_output_device"),
            size: rounded_max_output_size_bytes.try_into()?,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_host_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("ansi_output_host"),
            size: rounded_max_output_size_bytes.try_into()?,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encode_ansi"),
            });
        let texture_view = texture.create_view(&Default::default());

        self.calc_sizes_invoker.invoke(
            &mut encoder,
            [
                BindingResource::TextureView(&texture_view),
                offsets_buffer.as_entire_binding(),
            ],
            (texture.size().width, texture.size().height.div_ceil(2)),
        );
        self.prefix_sum_invoker.invoke(
            &mut encoder,
            [
                offsets_buffer.as_entire_binding(),
                total_size_device_buffer.as_entire_binding(),
            ],
            (1, 1),
        );
        self.encode_invoker.invoke(
            &mut encoder,
            [
                BindingResource::TextureView(&texture_view),
                offsets_buffer.as_entire_binding(),
                output_device_buffer.as_entire_binding(),
            ],
            (texture.size().width, texture.size().height.div_ceil(2)),
        );

        encoder.copy_buffer_to_buffer(
            &total_size_device_buffer,
            0,
            &total_size_host_buffer,
            0,
            total_size_host_buffer.size(),
        );
        encoder.copy_buffer_to_buffer(
            &output_device_buffer,
            0,
            &output_host_buffer,
            0,
            output_host_buffer.size(),
        );
        let index = self.queue.submit(Some(encoder.finish()));

        let (size_tx, size_rx) = tokio::sync::oneshot::channel();
        total_size_host_buffer.map_async(MapMode::Read, .., move |result| {
            size_tx.send(result).unwrap();
        });
        let (output_tx, output_rx) = tokio::sync::oneshot::channel();
        output_host_buffer.map_async(MapMode::Read, .., move |result| {
            output_tx.send(result).unwrap();
        });

        self.device.poll(wgpu::wgt::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })?;

        size_rx.await??;
        output_rx.await??;

        let size: usize = u32::from_le_bytes(
            total_size_host_buffer
                .get_mapped_range(..)
                .iter()
                .as_slice()
                .try_into()?,
        )
        .try_into()?;
        eprintln!("size = {} ({:#x})", size, size);
        let rounded_size = u64::try_from(size.div_ceil(4) * 4)?;
        let bytes = output_host_buffer.get_mapped_range(..rounded_size)[..size].to_vec();
        let s = unsafe { String::from_utf8_unchecked(bytes) };
        Ok(s)
    }
}

fn gen_pixels(width: usize, height: usize) -> Vec<u8> {
    let row: Vec<[u8; 4]> = (0..width * 2)
        .map(|n| {
            let hue = (n as f64) * 360.0f64 / ((width * 2) as f64);
            let (r, g, b) = hsv::hsv_to_rgb(hue, 1.0f64, 1.0f64);
            [r, g, b, 255]
        })
        .collect();
    (0..height)
        .flat_map(|n| {
            let off = n % width;
            row[off..off + width].iter()
        })
        .flatten()
        .copied()
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;
    let device = Rc::new(device);
    let queue = Rc::new(queue);

    const SIZE: (usize, usize) = (40, 30);
    let pixels: Vec<u8> = gen_pixels(SIZE.0, SIZE.1);
    let texture = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            label: Some("test_red"),
            size: wgpu::Extent3d {
                width: SIZE.0.try_into().unwrap(),
                height: SIZE.1.try_into().unwrap(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Uint,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba8Uint],
        },
        wgpu::wgt::TextureDataOrder::LayerMajor,
        &pixels,
    );

    let ansi_encoder = GpuAnsiEncoder::new(device, queue).await?;
    let s = ansi_encoder.ansi_from_texture(&texture).await?;
    println!("{s}\x1b[0m");

    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use std::rc::Rc;
    use wgpu::{
        BindingResource, BufferDescriptor, BufferUsages, Device, MapMode, Queue, TextureDescriptor,
        TextureFormat, TextureUsages, util::DeviceExt,
    };

    use crate::GpuFuncInvoker;

    async fn get_device() -> Result<(Rc<Device>, Rc<Queue>)> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;
        Ok((Rc::new(device), Rc::new(queue)))
    }

    async fn run_calc_sizes_test(
        device: &Device,
        queue: &Queue,
        invoker: &GpuFuncInvoker,
        pixels: &[u8],
        texture_size: (u32, u32),
    ) -> Result<Vec<u32>> {
        let texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("test_texture"),
                size: wgpu::Extent3d {
                    width: texture_size.0,
                    height: texture_size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba8Uint,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[TextureFormat::Rgba8Uint],
            },
            wgpu::wgt::TextureDataOrder::LayerMajor,
            pixels,
        );

        let num_outputs = texture_size.0 * ((texture_size.1 + 1) / 2);
        let sizes_buffer_size = (num_outputs * std::mem::size_of::<u32>() as u32) as u64;

        let sizes_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sizes_buffer"),
            size: sizes_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("staging_buffer"),
            size: sizes_buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });

        let texture_view = texture.create_view(&Default::default());
        invoker.invoke(
            &mut encoder,
            [
                BindingResource::TextureView(&texture_view),
                sizes_buffer.as_entire_binding(),
            ],
            (texture_size.0, (texture_size.1 + 1) / 2),
        );

        encoder.copy_buffer_to_buffer(&sizes_buffer, 0, &staging_buffer, 0, sizes_buffer_size);

        let submission_index = queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::wgt::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })?;

        rx.await??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        Ok(result)
    }

    #[tokio::test]
    async fn test_calc_sizes_even_width_height() -> Result<()> {
        let (device, queue) = get_device().await?;
        let calc_sizes_invoker = static_func_invoker!(device.clone(), "wgsl/calc_sizes.wgsl");

        let pixels_2x2: Vec<u8> = vec![
            255, 0, 0, 255, // Top-left: Red (255,0,0)
            0, 0, 255, 255, // Top-right: Blue (0,0,255)
            0, 255, 0, 255, // Bottom-left: Green (0,255,0)
            255, 255, 255, 255, // Bottom-right: White (255,255,255)
        ];
        let size_2x2 = (2, 2);
        let result_2x2 =
            run_calc_sizes_test(&device, &queue, &calc_sizes_invoker, &pixels_2x2, size_2x2)
                .await?;
        // Expected values calculated manually based on shader logic.
        // id(0,0): top(255,0,0), bot(0,255,0). len = 13 + (3+1+1) + 10 + (1+3+1) = 33
        // id(1,0): top(0,0,255), bot(255,255,255). len = 13 + (1+1+3) + 10 + (3+3+3) + 5 (eol) = 42
        assert_eq!(result_2x2, vec![33, 42]);
        Ok(())
    }

    #[tokio::test]
    async fn test_calc_sizes_even_width_odd_height() -> Result<()> {
        let (device, queue) = get_device().await?;
        let calc_sizes_invoker = static_func_invoker!(device.clone(), "wgsl/calc_sizes.wgsl");

        let pixels_2x1: Vec<u8> = vec![
            255, 0, 0, 255, // Top-left: Red (255,0,0)
            0, 0, 255, 255, // Top-right: Blue (0,0,255)
        ];
        let size_2x1 = (2, 1);
        let result_2x1 =
            run_calc_sizes_test(&device, &queue, &calc_sizes_invoker, &pixels_2x1, size_2x1)
                .await?;
        // Expected values:
        // id(0,0): top(255,0,0), no bot. len = 13 + (3+1+1) = 18
        // id(1,0): top(0,0,255), no bot. len = 13 + (1+1+3) + 5 (eol) = 23
        assert_eq!(result_2x1, vec![18, 23]);
        Ok(())
    }

    #[tokio::test]
    async fn test_calc_sizes_odd_width_height() -> Result<()> {
        let (device, queue) = get_device().await?;
        let calc_sizes_invoker = static_func_invoker!(device.clone(), "wgsl/calc_sizes.wgsl");

        let pixels_1x1: Vec<u8> = vec![
            10, 20, 30, 255, // (10,20,30)
        ];
        let size_1x1 = (1, 1);
        let result_1x1 =
            run_calc_sizes_test(&device, &queue, &calc_sizes_invoker, &pixels_1x1, size_1x1)
                .await?;

        // Expected values:
        // id(0,0): top(10,20,30), no bot. len = 13 + (2+2+2) + 5 (eol) = 24
        assert_eq!(result_1x1, vec![24]);
        Ok(())
    }

    #[tokio::test]
    async fn test_calc_sizes_multiple_of_4() -> Result<()> {
        let (device, queue) = get_device().await?;
        let calc_sizes_invoker = static_func_invoker!(device.clone(), "wgsl/calc_sizes.wgsl");

        let pixels_2x4: Vec<u8> = vec![
            255, 0, 0, 255, // (0, 0)
            0, 255, 0, 255, // (0, 1)
            0, 0, 11, 255,  // (1, 0)
            1, 0, 0, 255,   // (1, 1)
            255, 0, 0, 255, // (0, 2)
            0, 255, 0, 255, // (0, 3)
            0, 0, 11, 255,  // (1, 2)
            1, 0, 0, 255,   // (1, 3)
        ];
        let size_2x4 = (2, 4);
        let result_2x4 =
            run_calc_sizes_test(&device, &queue, &calc_sizes_invoker, &pixels_2x4, size_2x4)
                .await?;
        // Expected values calculated manually based on shader logic.
        // id(0,0): top(255,0,0), bot(0,0,11). len = 13 + (3+1+1) + 10 + (1+1+2) = 32
        // id(1,0): top(0,0,255), bot(1,0,0). len = 13 + (1+1+3) + 10 + (1+1+1) + 5 (eol) = 36
        // id(0,1): top(255,0,0), bot(0,0,11). len = 13 + (3+1+1) + 10 + (1+1+2) = 32
        // id(1,1): top(0,0,255), bot(1,0,0). len = 13 + (1+1+3) + 10 + (1+1+1) + 5 (eol) = 36
        assert_eq!(result_2x4, vec![32, 36, 32, 36]);
        Ok(())
    }

    async fn run_encode_ansi_test(
        device: &Device,
        queue: &Queue,
        invoker: &GpuFuncInvoker,
        pixels: &[u8],
        texture_size: (u32, u32),
        offsets: &[u32],
    ) -> Result<String> {
        let texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("test_texture"),
                size: wgpu::Extent3d {
                    width: texture_size.0,
                    height: texture_size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba8Uint,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[TextureFormat::Rgba8Uint],
            },
            wgpu::wgt::TextureDataOrder::LayerMajor,
            pixels,
        );

        let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offsets_buffer"),
            contents: bytemuck::cast_slice(offsets),
            usage: BufferUsages::STORAGE,
        });

        let total_size = offsets.iter().sum::<u32>() as u64;
        let output_buffer_size = total_size.max(1);
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("output_buffer"),
            size: output_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("staging_buffer"),
            size: output_buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });

        let texture_view = texture.create_view(&Default::default());
        invoker.invoke(
            &mut encoder,
            [
                BindingResource::TextureView(&texture_view),
                offsets_buffer.as_entire_binding(),
                output_buffer.as_entire_binding(),
            ],
            (texture_size.0, (texture_size.1 + 1) / 2),
        );

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);

        let submission_index = queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::wgt::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })?;

        rx.await??;

        let data = buffer_slice.get_mapped_range();
        let result = String::from_utf8_lossy(&data).to_string();

        Ok(result)
    }

    #[tokio::test]
    async fn test_encode_ansi() -> Result<()> {
        let (device, queue) = get_device().await?;
        let encode_ansi_invoker = static_func_invoker!(device.clone(), "wgsl/encode_ansi.wgsl");

        let pixels: Vec<u8> = vec![
            255, 0, 0, 255, // Top-left: Red
            0, 0, 255, 255, // Top-right: Blue
            0, 255, 0, 255, // Bottom-left: Green
            255, 255, 255, 255, // Bottom-right: White
        ];
        let texture_size = (2, 2);
        let offsets = &[0, 33];
        let result =
            run_encode_ansi_test(&device, &queue, &encode_ansi_invoker, &pixels, texture_size, offsets)
                .await?;

        let expected = "\x1b[38;2;255;0;0m\x1b[48;2;0;255;0m\u{2580}\x1b[38;2;0;0;255m\x1b[48;2;255;255;255m\u{2580}\x1b[0m\n";
        assert_eq!(result.len(), expected.len());
        assert_eq!(result, expected);

        Ok(())
    }
}
