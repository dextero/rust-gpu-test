use anyhow::Result;
use pin_project::pin_project;
use std::rc::Rc;
use std::task::Poll;
use wgpu::BindGroupDescriptor;
use wgpu::BindGroupEntry;
use wgpu::BindingResource;
use wgpu::Buffer;
use wgpu::BufferDescriptor;
use wgpu::BufferUsages;
use wgpu::BufferView;
use wgpu::ComputePipeline;
use wgpu::Device;
use wgpu::MapMode;
use wgpu::PollType;
use wgpu::Queue;
use wgpu::ShaderModule;
use wgpu::ShaderModuleDescriptor;
use wgpu::ShaderSource;
use wgpu::SubmissionIndex;
use wgpu::Texture;
use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;
use wgpu::wgt::CommandEncoderDescriptor;

#[pin_project]
struct BufferMapFuture<'d, InnerFut: Future<Output = Result<()>>> {
    device: &'d Device,
    buffer: Buffer,
    #[pin]
    inner: InnerFut,
}

impl<'d, InnerFut> Future for BufferMapFuture<'d, InnerFut>
where
    InnerFut: Future<Output = Result<()>>,
{
    type Output = Result<Vec<u8>>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        match self.device.poll(PollType::Poll) {
            Ok(_) => {}
            Err(e) => return Poll::Ready(Err(e.into())),
        }

        let project = self.project();
        match project.inner.poll(cx) {
            Poll::Ready(val) => Poll::Ready(Ok(project.buffer.get_mapped_range(..).to_vec())),
            Poll::Pending => Poll::Pending,
        }
    }
}

fn async_map(device: &Device, buffer: Buffer) -> impl Future<Output = Result<Vec<u8>>> + '_ {
    let (tx, rx) = tokio::sync::oneshot::channel();
    buffer.map_async(MapMode::Read, .., move |result| {
        tx.send(result).unwrap();
    });
    BufferMapFuture {
        device,
        buffer,
        inner: async { Result::Ok(rx.await??) },
    }
}

struct GpuFunc {
    device: Rc<Device>,
    pipeline: ComputePipeline,
}

impl GpuFunc {
    pub(crate) fn from_shader(device: Rc<Device>, shader: &ShaderModule, entry: &str) -> Self {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", entry)),
            layout: None,
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { device, pipeline }
    }
}

struct CalcSizes(GpuFunc);

impl CalcSizes {
    pub fn call(&self, queue: &Queue, texture: &Texture) -> Result<(SubmissionIndex, Buffer)> {
        let mut encoder = self
            .0
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("calc_sizes"),
            });
        let texture_size = texture.size();
        let num_pixels = texture_size.width * texture_size.height;
        let offsets_size_bytes = usize::try_from(num_pixels).unwrap() * std::mem::size_of::<u32>();
        let offsets_buffer = self.0.device.create_buffer(&BufferDescriptor {
            label: Some("sizes_offsets"),
            size: offsets_size_bytes.try_into().unwrap(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.0.device.create_bind_group(&BindGroupDescriptor {
            label: Some("calc_sizes_bind_group"),
            layout: &self.0.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &texture.create_view(&Default::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: offsets_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("calc_sizes"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.0.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_pixels, 1, 1);
        }

        Ok((queue.submit(Some(encoder.finish())), offsets_buffer))
    }
}

struct PrefixSum(GpuFunc);

impl PrefixSum {
    pub fn call(
        &self,
        queue: &Queue,
        buffer: &Buffer,
        input_stride: u32,
    ) -> Result<SubmissionIndex> {
        let mut encoder = self
            .0
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("prefix_sum"),
            });
        let buffer_size_elems = buffer.size() / u64::try_from(std::mem::size_of::<u32>()).unwrap();
        let num_threads = (buffer_size_elems / u64::from(input_stride)).min(128 * 256);
        let stride_buffer = self.0.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("input_stride_buffer"),
            contents: &input_stride.to_le_bytes(),
            usage: BufferUsages::STORAGE,
        });
        let bind_group = self.0.device.create_bind_group(&BindGroupDescriptor {
            label: Some("prefix_sum_bind_group"),
            layout: &self.0.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: stride_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_sum"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.0.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(u32::try_from(num_threads).unwrap(), 1, 1);
        }

        Ok(queue.submit(Some(encoder.finish())))
    }
}

struct AddPartialSums(GpuFunc);

impl AddPartialSums {
    pub fn call(&self, queue: &Queue, buffer: &Buffer) -> Result<(SubmissionIndex, Buffer)> {
        let mut encoder = self
            .0
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("add_partial_sums"),
            });
        let buffer_size_elems = buffer.size() / u64::try_from(std::mem::size_of::<u32>()).unwrap();
        let total_size = self.0.device.create_buffer(&BufferDescriptor {
            label: Some("total_size_buffer"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            size: std::mem::size_of::<u32>().try_into().unwrap(),
            mapped_at_creation: false,
        });
        let bind_group = self.0.device.create_bind_group(&BindGroupDescriptor {
            label: Some("add_partial_sums"),
            layout: &self.0.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: total_size.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("add_partial_sums"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.0.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(u32::try_from(buffer_size_elems).unwrap(), 1, 1);
        }

        Ok((queue.submit(Some(encoder.finish())), total_size))
    }
}

struct EncodeAnsi(GpuFunc);

impl EncodeAnsi {
    pub fn call(
        &self,
        queue: &Queue,
        texture: &Texture,
        offsets: &Buffer,
        total_size_device: &Buffer,
    ) -> Result<(SubmissionIndex, Buffer, Buffer)> {
        let texture_size = texture.size();
        let num_pixels = texture_size.width * texture_size.height;
        let total_size_host_buffer = self.0.device.create_buffer(&BufferDescriptor {
            label: Some("ansi_output_host"),
            size: std::mem::size_of::<u32>().try_into()?,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let max_output_size_bytes: usize = usize::try_from(num_pixels)?
            * "\x1b[38;2;255;255;255m\x1b[48;2;255;255;255m\u{2580}".len()
            + usize::try_from(texture_size.height)? * "\x1b[0m\n".len();
        let rounded_max_output_size_bytes = max_output_size_bytes.div_ceil(4) * 4;
        let output_device_buffer = self.0.device.create_buffer(&BufferDescriptor {
            label: Some("ansi_output_device"),
            size: rounded_max_output_size_bytes.try_into()?,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_host_buffer = self.0.device.create_buffer(&BufferDescriptor {
            label: Some("ansi_output_host"),
            size: rounded_max_output_size_bytes.try_into()?,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .0
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("encode_ansi"),
            });
        let num_pixels = (texture.size().width * texture.size().height).min(128 * 256);
        let bind_group = self.0.device.create_bind_group(&BindGroupDescriptor {
            label: Some("encode_ansi_bind_group"),
            layout: &self.0.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &texture.create_view(&Default::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: offsets.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_device_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("encode_ansi"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.0.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_pixels, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &total_size_device,
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

        Ok((
            queue.submit(Some(encoder.finish())),
            total_size_host_buffer,
            output_host_buffer,
        ))
    }
}

pub struct GpuAnsiEncoder {
    device: Rc<Device>,
    queue: Rc<Queue>,
    calc_sizes: CalcSizes,
    prefix_sum: PrefixSum,
    add_partial_sums: AddPartialSums,
    encode_ansi: EncodeAnsi,
}

impl GpuAnsiEncoder {
    pub async fn new(device: Rc<Device>, queue: Rc<Queue>) -> Result<Self> {
        let calc_sizes_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("wgsl/calc_sizes.wgsl"),
            source: ShaderSource::Wgsl(include_str!("wgsl/calc_sizes.wgsl").into()),
        });
        let calc_sizes = CalcSizes(GpuFunc::from_shader(
            device.clone(),
            &calc_sizes_shader,
            "calc_sizes",
        ));

        let prefix_sum_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("wgsl/prefix_sum.wgsl"),
            source: ShaderSource::Wgsl(include_str!("wgsl/prefix_sum.wgsl").into()),
        });
        let prefix_sum = PrefixSum(GpuFunc::from_shader(
            device.clone(),
            &prefix_sum_shader,
            "prefix_sum",
        ));

        let add_partial_sums_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("wgsl/add_partial_sums.wgsl"),
            source: ShaderSource::Wgsl(include_str!("wgsl/add_partial_sums.wgsl").into()),
        });
        let add_partial_sums = AddPartialSums(GpuFunc::from_shader(
            device.clone(),
            &add_partial_sums_shader,
            "add_partial_sums",
        ));

        let encode_ansi_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("encode_ansi"),
            source: ShaderSource::Wgsl(include_str!("wgsl/encode_ansi.wgsl").into()),
        });
        let encode_ansi = EncodeAnsi(GpuFunc::from_shader(
            device.clone(),
            &encode_ansi_shader,
            "encode_ansi",
        ));

        Ok(Self {
            device,
            queue,
            calc_sizes,
            prefix_sum,
            add_partial_sums,
            encode_ansi,
        })
    }

    pub async fn ansi_from_texture(&self, texture: &Texture) -> Result<String> {
        let texture_size = texture.size();
        let num_pixels = texture_size.width * texture_size.height;

        let (_, offsets) = self.calc_sizes.call(&self.queue, texture)?;
        let mut stride = 1;
        while stride < num_pixels {
            let _ = self.prefix_sum.call(&self.queue, &offsets, stride)?;
            stride *= 256;
        }
        let (_, total_size_device) = self.add_partial_sums.call(&self.queue, &offsets)?;
        let (submission_index, total_size_host_buffer, output_host_buffer) = self
            .encode_ansi
            .call(&self.queue, texture, &offsets, &total_size_device)?;

        let total_size_fut = async {
            let bytes = async_map(&self.device, total_size_host_buffer).await?;
            let u32s: &[u32] = bytemuck::cast_slice(bytes.as_slice());
            Result::<u32>::Ok(u32s[0])
        };
        let output_fut = async {
            let mut bytes = async_map(&self.device, output_host_buffer).await?;
            bytes.shrink_to(usize::try_from(total_size_fut.await?)?);
            Result::<String>::Ok(unsafe { String::from_utf8_unchecked(bytes) })
        };

        self.device.poll(wgpu::wgt::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })?;

        let s = unsafe { String::from_utf8_unchecked(output_fut.await?.into()) };
        Ok(s)
    }
}

pub fn gen_pixels(width: usize, height: usize) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use insta::assert_debug_snapshot;
    use std::rc::Rc;
    use wgpu::{
        BufferDescriptor, BufferUsages, Device, MapMode, Queue, ShaderModuleDescriptor,
        ShaderSource, TextureDescriptor, TextureFormat, TextureUsages,
        util::{BufferInitDescriptor, DeviceExt},
    };

    use crate::gpu_ansi_encoder::{CalcSizes, EncodeAnsi, GpuFunc, PrefixSum, async_map};

    async fn get_device() -> Result<(Rc<Device>, Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;
        Ok((Rc::new(device), queue))
    }

    async fn run_calc_sizes_test(pixels: &[u8], texture_size: (u32, u32)) -> Result<Vec<u32>> {
        let (device, queue) = get_device().await?;
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("wgsl/calc_sizes.wgsl").into()),
        });
        let calc_sizes = CalcSizes(GpuFunc::from_shader(device.clone(), &shader, "calc_sizes"));

        let texture = device.create_texture_with_data(
            &queue,
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

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("staging_buffer"),
            size: sizes_buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });

        let (_, sizes_buffer) = calc_sizes.call(&queue, &texture)?;

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
        assert_debug_snapshot!(
            run_calc_sizes_test(
                &[
                    255, 0, 0, 255, // "\x1b[38;2;255;0;0m"     - seq 0, 10+3+1+1=15
                    0, 0, 255, 255, // "\x1b[38;2;0;0;255m"     - seq 1, 10+1+1+3=15
                    0, 255, 0, 255, // "\x1b[48;2;0;255;0"      - seq 0, 10+1+3+1=15
                    255, 255, 255, 255, // "\x1b[48;2;255;255;255m" - seq 1, 10+3+3+3=19
                ],
                // +3 per UTF-8 character; +5 per "\x1b[0m\n" @ EOL
                (2, 2)
            )
            .await?
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_calc_sizes_even_width_odd_height() -> Result<()> {
        assert_debug_snapshot!(
            run_calc_sizes_test(
                &[
                    255, 0, 0, 255, // "\x1b[38;2;255;0;0m"   - seq 0, 10+3+1+1=15
                    0, 255, 0, 255, // "\x1b[48;2;255;0;255m" - seq 1, 10+3+1+3=17
                ],
                // +3 per UTF-8 character; +5 per "\x1b[0m\n" @ EOL
                (2, 1)
            )
            .await?
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_calc_sizes_odd_width_height() -> Result<()> {
        // "\x1b[38;2;10;20;30m\xe2\x96\x80\x1b[0m\n" = 10+2+2+2+3+5=24
        assert_debug_snapshot!(run_calc_sizes_test(&[10, 20, 30, 255], (1, 1)).await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_calc_sizes_multiple_of_4() -> Result<()> {
        assert_debug_snapshot!(
            run_calc_sizes_test(
                &[
                    255, 0, 0, 255, // (0, 0)
                    0, 255, 0, 255, // (0, 1)
                    0, 0, 11, 255, // (1, 0)
                    1, 0, 0, 255, // (1, 1)
                    255, 0, 0, 255, // (0, 2)
                    0, 255, 0, 255, // (0, 3)
                    0, 0, 11, 255, // (1, 2)
                    1, 0, 0, 255, // (1, 3)
                ],
                (2, 4)
            )
            .await?
        );
        Ok(())
    }

    async fn run_prefix_sum_test(sizes: &[u32], stride: u32) -> Result<Vec<u32>> {
        let (device, queue) = get_device().await?;
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("wgsl/prefix_sum.wgsl").into()),
        });
        let prefix_sum = PrefixSum(GpuFunc::from_shader(device.clone(), &shader, "prefix_sum"));

        let sizes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sizes_buffer"),
            contents: bytemuck::cast_slice(sizes),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let sizes_staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sizes_staging_buffer"),
            size: sizes_buffer.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });

        prefix_sum.call(&queue, &sizes_buffer, stride)?;

        encoder.copy_buffer_to_buffer(
            &sizes_buffer,
            0,
            &sizes_staging_buffer,
            0,
            sizes_buffer.size(),
        );

        let submission_index = queue.submit(Some(encoder.finish()));

        let sizes_slice = sizes_staging_buffer.slice(..);
        let (sizes_tx, sizes_rx) = tokio::sync::oneshot::channel();
        sizes_slice.map_async(MapMode::Read, move |result| {
            sizes_tx.send(result).unwrap();
        });

        device.poll(wgpu::wgt::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })?;

        sizes_rx.await??;

        let sizes_data = sizes_slice.get_mapped_range();
        let offsets: Vec<u32> = bytemuck::cast_slice(&sizes_data).to_vec();

        Ok(offsets)
    }

    #[tokio::test]
    async fn test_prefix_sum() -> Result<()> {
        assert_debug_snapshot!(run_prefix_sum_test(&[10, 20, 5, 30, 15], 1).await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_prefix_sum_with_stride() -> Result<()> {
        let sparse_ones = vec![1u32; 16];
        assert_debug_snapshot!(run_prefix_sum_test(&sparse_ones, 2).await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_prefix_sum_512_with_stride_128() -> Result<()> {
        let ones = vec![1u32; 512];
        let result = run_prefix_sum_test(&ones, 128).await?;
        assert_debug_snapshot!(result.iter().skip(127).step_by(128).collect::<Vec<_>>());
        for n in 0..result.len() {
            if n % 128 != 127 {
                assert_eq!(result[n], 1, "at index {n}");
            }
        }
        Ok(())
    }

    async fn run_encode_ansi_test(
        pixels: &[u8],
        texture_size: (u32, u32),
        sizes: &[u32],
    ) -> Result<String> {
        let (device, queue) = get_device().await?;
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("wgsl/encode_ansi.wgsl").into()),
        });
        let encode_ansi = EncodeAnsi(GpuFunc::from_shader(device.clone(), &shader, "encode_ansi"));

        let mut offsets = Vec::with_capacity(sizes.len());
        let mut current_offset = 0;
        for size in sizes {
            current_offset += *size;
            offsets.push(current_offset);
        }
        let total_size = current_offset;

        let texture = device.create_texture_with_data(
            &queue,
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
            contents: bytemuck::cast_slice(&offsets),
            usage: BufferUsages::STORAGE,
        });
        let total_size_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("total_size"),
            contents: bytemuck::cast_slice(&total_size.to_le_bytes()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let (submission_index, total_size_buffer, output_buffer) =
            encode_ansi.call(&queue, &texture, &offsets_buffer, &total_size_buffer)?;

        let total_size_fut = async {
            let bytes = async_map(&device, total_size_buffer).await?;
            let u32s: &[u32] = bytemuck::cast_slice(bytes.as_slice());
            Result::<u32>::Ok(u32s[0])
        };
        let output_fut = async {
            let bytes = async_map(&device, output_buffer).await?;
            let slice = &bytes[..usize::try_from(total_size_fut.await?)?];
            Result::<String>::Ok(String::from_utf8_lossy(slice).into_owned())
        };

        device.poll(wgpu::wgt::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })?;

        Ok(output_fut.await?)
    }

    #[tokio::test]
    async fn test_encode_ansi() -> Result<()> {
        assert_debug_snapshot!(
            run_encode_ansi_test(
                &[
                    255, 0, 0, 255, // (0, 0)
                    0, 0, 255, 255, // (1, 0)
                    0, 255, 0, 255, // (0, 1)
                    255, 255, 255, 255, // (1, 1)
                ],
                (2, 2),
                &[33, 42]
            )
            .await?
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_encode_ansi_multiple_of_4() -> Result<()> {
        assert_debug_snapshot!(
            run_encode_ansi_test(
                &[
                    255, 0, 0, 255, // (0, 0)
                    0, 255, 0, 255, // (1, 0)
                    0, 0, 11, 255, // (0, 1)
                    1, 0, 0, 255, // (1, 1)
                    255, 0, 0, 255, // (0, 2)
                    0, 255, 0, 255, // (1, 2)
                ],
                (2, 3),
                &[32, 36, 18, 23]
            )
            .await?
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_encode_ascii_zero_in_the_middle() -> Result<()> {
        assert_debug_snapshot!(
            run_encode_ansi_test(
                &[
                    200, 200, 200, 255, //
                    200, 200, 200, 255, //
                    200, 200, 200, 255, //
                    200, 200, 200, 255, //
                ],
                (2, 2),
                &[41, 46]
            )
            .await?
        );
        Ok(())
    }

    const RAINBOW_4X3: &[u8] = &[
        255, 0, 0, 255, //   15
        255, 191, 0, 255, // 17
        127, 255, 0, 255, // 17
        0, 255, 63, 255, //  16
        255, 191, 0, 255, //   +17+3  =35
        127, 255, 0, 255, //   +17+3  =37
        0, 255, 63, 255, //    +16+3  =36
        0, 255, 255, 255, //   +17+3+5=41
        127, 255, 0, 255, // 17+3     =20
        0, 255, 63, 255, //  16+3     =19
        0, 255, 255, 255, // 17+3     =20
        0, 63, 255, 255, //  16+3+5   =24
    ];

    #[tokio::test]
    async fn test_rainbow_4x3_calc_sizes() -> Result<()> {
        assert_debug_snapshot!(run_calc_sizes_test(RAINBOW_4X3, (4, 3)).await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_rainbow_4x3_prefix_sum() -> Result<()> {
        assert_debug_snapshot!(run_prefix_sum_test(&[35, 37, 36, 41, 20, 19, 20, 24], 1).await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_rainbow_4x3_encode_ansi() -> Result<()> {
        assert_debug_snapshot!(
            run_encode_ansi_test(&RAINBOW_4X3, (4, 3), &[35, 37, 36, 41, 20, 19, 20, 24]).await?
        );
        Ok(())
    }
}
