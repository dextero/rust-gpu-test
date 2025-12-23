use anyhow::Result;
use std::{borrow::Cow, rc::Rc};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, BufferDescriptor, BufferUsages,
    ComputePipeline, Device, MapMode, Queue, ShaderModuleDescriptor, ShaderSource, Texture,
    TextureFormat, TextureUsages, util::DeviceExt, wgt::TextureDescriptor,
};

struct GpuAnsiEncoder {
    device: Rc<Device>,
    queue: Rc<Queue>,
    calc_sizes_pipeline: ComputePipeline,
    prefix_sum_pipeline: ComputePipeline,
    encode_pipeline: ComputePipeline,
}

impl GpuAnsiEncoder {
    pub async fn new(device: Rc<Device>, queue: Rc<Queue>) -> Result<Self> {
        macro_rules! get_pipeline {
            ($src:literal) => {{
                let shader = device.create_shader_module(ShaderModuleDescriptor {
                    label: Some($src),
                    source: ShaderSource::Wgsl(Cow::Borrowed(include_str!($src))),
                });

                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some($src),
                    layout: None,
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                })
            }};
        }

        let calc_sizes_pipeline = get_pipeline!("wgsl/calc_sizes.wgsl");
        let prefix_sum_pipeline = get_pipeline!("wgsl/prefix_sum.wgsl");
        let encode_pipeline = get_pipeline!("wgsl/encode_ansi.wgsl");

        Ok(Self {
            device,
            queue,
            calc_sizes_pipeline,
            prefix_sum_pipeline,
            encode_pipeline,
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
            * "\x1b[38;2;255;255;255m\x1b[48;2;255;255;255m\u{2580}"
                .as_bytes()
                .len();
        let output_device_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("ansi_output_device"),
            size: max_output_size_bytes.try_into()?,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_host_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("ansi_output_host"),
            size: max_output_size_bytes.try_into()?,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encode_ansi"),
            });
        let texture_view = texture.create_view(&Default::default());

        {
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("calc_sizes_bind_group"),
                layout: &self.calc_sizes_pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: offsets_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("calc_sizes"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.calc_sizes_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(texture.size().width, (texture.size().height + 1) / 2, 1);
        }
        {
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("prefix_sum_bind_group"),
                layout: &self.prefix_sum_pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: offsets_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: total_size_device_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_sum"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.prefix_sum_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("encode_ansi_bind_group"),
                layout: &self.encode_pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: offsets_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: output_device_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("encode_ansi"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.encode_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                texture.size().width,
                (texture.size().height + 1) / 2,
                1,
            );
        }

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
        ).try_into()?;
        eprintln!("size = {} ({:#x})", size, size);
        let rounded_size = u64::try_from(((size + 3) / 4) * 4)?;
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
            row[off..off + width].into_iter()
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
    println!("{s}");

    Ok(())
}
