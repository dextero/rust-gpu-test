use anyhow::Result;
use std::{borrow::Cow, rc::Rc};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline, Device, MapMode, Queue, ShaderModule, ShaderModuleDescriptor, ShaderSource, Texture, TextureFormat, TextureUsages, util::DeviceExt, wgt::TextureDescriptor
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
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ansi_encoder"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let get_pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let calc_sizes_pipeline = get_pipeline("calc_sizes");
        let prefix_sum_pipeline = get_pipeline("prefix_sum");
        let encode_pipeline = get_pipeline("encode");

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
        let mut offsets_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("sizes_offsets"),
            size: offsets_size_bytes.try_into()?,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let max_output_size_bytes: usize = usize::try_from(num_pixels)?
            * "\x1b[38;2;255;255;255m\x1b48;2;255;255;255m\u{2580}"
                .as_bytes()
                .len();
        let output_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("ansi_output"),
            size: max_output_size_bytes.try_into()?,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encode_ansi"),
            });

        self.calc_sizes(&mut encoder, texture, &mut offsets_buffer)?;
        self.prefix_sum(&mut encoder, &mut offsets_buffer)?;
        self.encode(encoder, texture, &offsets_buffer, output_buffer)
            .await
    }

    fn calc_sizes(
        &self,
        encoder: &mut CommandEncoder,
        texture: &Texture,
        offsets_buffer: &mut Buffer,
    ) -> Result<()> {
        let texture_view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.calc_sizes_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
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
        compute_pass.dispatch_workgroups(texture.size().width, texture.size().height, 1);

        Ok(())
    }

    fn prefix_sum(&self, encoder: &mut CommandEncoder, offsets: &mut Buffer) -> Result<()> {
        let num_offsets = usize::try_from(offsets.size())? * std::mem::size_of::<u32>();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.calc_sizes_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 1,
                resource: offsets.as_entire_binding(),
            }],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prefix_sum"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.prefix_sum_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(num_offsets.try_into()?, 1, 1);

        Ok(())
    }

    async fn encode(
        &self,
        mut encoder: CommandEncoder,
        texture: &Texture,
        offsets: &Buffer,
        output: Buffer,
    ) -> Result<String> {
        let num_offsets = usize::try_from(offsets.size())? * std::mem::size_of::<u32>();
        let texture_view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.calc_sizes_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("encode_ansi"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.encode_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_offsets.try_into()?, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = tokio::sync::oneshot::channel();
        output.map_async(MapMode::Read, .., move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::wgt::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        receiver.await??;

        // TODO: check bounds
        let bytes = output.get_mapped_range(..).to_vec();
        let s = unsafe { String::from_utf8_unchecked(bytes) };
        Ok(s)
    }
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

    let pixels: &[u8] = &[255, 0, 0, 0];
    let texture = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            label: Some("test_red"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
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

    let ansi_encoder = GpuAnsiEncoder::new(device, queue).await?;
    let s = ansi_encoder.ansi_from_texture(&texture).await?;
    println!("{s}");

    Ok(())
}
