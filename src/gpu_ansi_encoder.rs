use anyhow::Result;
use pin_project::pin_project;
use zerocopy::IntoByteSlice;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::Poll;
use std::task::Waker;
use wgpu::BindGroupDescriptor;
use wgpu::BindGroupEntry;
use wgpu::BindingResource;
use wgpu::Buffer;
use wgpu::BufferDescriptor;
use wgpu::BufferUsages;
use wgpu::CommandEncoderDescriptor;
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
use zerocopy::Immutable;
use zerocopy::TryFromBytes;

struct InspectableBuffer {
    device_buffer: Buffer,
    host_buffer: Buffer,
}

impl InspectableBuffer {
    fn new(device: &Device, label: &'static str, size: usize) -> Result<Self> {
        let label = Some(label);
        let size = size.try_into()?;
        let device_buffer = device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let host_buffer = device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(Self {
            device_buffer,
            host_buffer,
        })
    }

    #[cfg(test)]
    fn new_with_data(device: &Device, label: &'static str, contents: &[u8]) -> Result<Self> {
        let label = Some(label);
        let device_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label,
            contents,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let host_buffer = device.create_buffer(&BufferDescriptor {
            label,
            size: contents.len().try_into()?,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(Self {
            device_buffer,
            host_buffer,
        })
    }

    async fn read<'a, T: Immutable + Copy + 'static>(
        &'a self,
        device: &Device,
        queue: &Queue,
    ) -> Result<Vec<T>>
    where
        [T]: TryFromBytes,
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.device_buffer,
            0,
            &self.host_buffer,
            0,
            self.host_buffer.size(),
        );
        queue.submit(Some(encoder.finish()));
        let bytes = async_map::<T>(device, self.host_buffer.clone()).await?;
        Ok(bytes)
    }

    fn size(&self) -> u64 {
        self.device_buffer.size()
    }

    fn as_entire_binding(&self) -> BindingResource<'_> {
        self.device_buffer.as_entire_binding()
    }
}

#[pin_project]
struct BufferMapFuture<'d, InnerFut: Future<Output = Result<()>>, T: Immutable + Clone + 'static>
where
    [T]: TryFromBytes,
{
    device: &'d Device,
    buffer: Buffer,
    #[pin]
    inner: InnerFut,
    waker: Arc<Mutex<Option<Waker>>>,
    _t: PhantomData<T>,
}

impl<'d, InnerFut, T: Immutable + Clone + 'static> Future for BufferMapFuture<'d, InnerFut, T>
where
    InnerFut: Future<Output = Result<()>>,
    [T]: TryFromBytes,
{
    type Output = Result<Vec<T>>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        match self.device.poll(PollType::Poll) {
            Ok(_) => {}
            Err(e) => return Poll::Ready(Err(e.into())),
        }

        let project = self.project();
        match project.inner.poll(cx) {
            Poll::Ready(_) => {
                let size = usize::try_from(project.buffer.size()).unwrap();
                if size % std::mem::size_of::<T>() != 0 {
                    panic!(
                        "cannot read buffer of size {}, required alignment = {}",
                        project.buffer.size(),
                        std::mem::size_of::<T>()
                    );
                }
                let view = project.buffer.get_mapped_range(..).to_vec();
                // TODO WTF
                let slice: &'static [u8] = unsafe { std::mem::transmute(view.as_slice()) };
                let v = <[T]>::try_ref_from_bytes(slice)?;
                let v = v.into_iter().cloned().collect();
                Poll::Ready(Ok(v))
            }
            Poll::Pending => {
                project.waker.lock().unwrap().replace(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

fn async_map<T: Immutable + Clone + 'static>(
    device: &Device,
    buffer: Buffer,
) -> impl Future<Output = Result<Vec<T>>> + '_
where
    [T]: TryFromBytes,
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    let waker: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
    let waker_clone = waker.clone();
    buffer.map_async(MapMode::Read, .., move |result| {
        tx.send(result).unwrap();
        if let Some(waker) = waker_clone.lock().unwrap().take() {
            waker.wake();
        }
    });
    BufferMapFuture {
        device,
        buffer,
        inner: async { Result::Ok(rx.await??) },
        waker,
        _t: PhantomData,
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
    pub fn call(
        &self,
        queue: &Queue,
        texture: &Texture,
    ) -> Result<(SubmissionIndex, InspectableBuffer)> {
        let mut encoder = self
            .0
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("calc_sizes"),
            });
        let texture_size = texture.size();
        let num_pixels = texture_size.width * texture_size.height;
        let offsets_size_bytes = usize::try_from(num_pixels).unwrap() * std::mem::size_of::<u32>();
        let offsets_buffer =
            InspectableBuffer::new(&self.0.device, "sizes_offsets", offsets_size_bytes)?;

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
        buffer: &InspectableBuffer,
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
    pub fn call(
        &self,
        queue: &Queue,
        buffer: &InspectableBuffer,
    ) -> Result<(SubmissionIndex, InspectableBuffer)> {
        let mut encoder = self
            .0
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("add_partial_sums"),
            });
        let buffer_size_elems = buffer.size() / u64::try_from(std::mem::size_of::<u32>()).unwrap();
        let total_size = InspectableBuffer::new(
            &self.0.device,
            "total_size_buffer",
            std::mem::size_of::<u32>(),
        )?;
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
        offsets: &InspectableBuffer,
    ) -> Result<(SubmissionIndex, InspectableBuffer)> {
        let texture_size = texture.size();
        let num_pixels = texture_size.width * texture_size.height;
        let max_output_size_bytes: usize = usize::try_from(num_pixels)?
            * "\x1b[38;2;255;255;255m\x1b[48;2;255;255;255m\u{2580}".len()
            + usize::try_from(texture_size.height)? * "\x1b[0m\n".len();
        let rounded_max_output_size_bytes = max_output_size_bytes.div_ceil(4) * 4;
        let output_buffer = InspectableBuffer::new(
            &self.0.device,
            "ansi_output_device",
            rounded_max_output_size_bytes,
        )?;
        dbg!(max_output_size_bytes);

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
                    resource: output_buffer.as_entire_binding(),
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

        Ok((queue.submit(Some(encoder.finish())), output_buffer))
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
        let (_, total_size_buffer) = self.add_partial_sums.call(&self.queue, &offsets)?;
        let (_, output_buffer) = self.encode_ansi.call(&self.queue, texture, &offsets)?;
        self.device.poll(PollType::wait_indefinitely())?;

        let size = total_size_buffer
            .read::<u32>(&self.device, &self.queue)
            .await?[0];
        let mut output = output_buffer.read::<u8>(&self.device, &self.queue).await?;
        output.shrink_to(usize::try_from(size)?);
        let s = unsafe { String::from_utf8_unchecked(output) };

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
    use itertools::Itertools;
    use std::rc::Rc;
    use wgpu::{
        Device, PollType, Queue, ShaderModuleDescriptor, ShaderSource, TextureDescriptor,
        TextureFormat, TextureUsages, util::DeviceExt,
    };
    use zerocopy::IntoBytes;

    use crate::gpu_ansi_encoder::{
        AddPartialSums, CalcSizes, EncodeAnsi, GpuFunc, InspectableBuffer, PrefixSum,
    };

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

        let (_, sizes_buffer) = calc_sizes.call(&queue, &texture)?;
        device.poll(PollType::wait_indefinitely())?;

        Ok(sizes_buffer.read::<u32>(&device, &queue).await?)
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

        let sizes_buffer =
            InspectableBuffer::new_with_data(&device, "sizes_buffer", sizes.as_bytes())?;

        prefix_sum.call(&queue, &sizes_buffer, stride)?;
        device.poll(PollType::wait_indefinitely())?;

        let offsets = sizes_buffer.read::<u32>(&device, &queue).await?;
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

        let offsets_buffer =
            InspectableBuffer::new_with_data(&device, "offsets_buffer", offsets.as_bytes())?;

        let (_, output_buffer) = encode_ansi.call(&queue, &texture, &offsets_buffer)?;
        device.poll(PollType::wait_indefinitely())?;

        let mut output = output_buffer.read::<u8>(&device, &queue).await?;
        output.shrink_to(usize::try_from(total_size)?);
        let s = unsafe { String::from_utf8_unchecked(output) };
        Ok(s)
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

    async fn run_add_partial_sums_test(values: &[u32]) -> Result<(Vec<u32>, u32)> {
        let (device, queue) = get_device().await?;
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("wgsl/add_partial_sums.wgsl").into()),
        });
        let add_partial_sums = AddPartialSums(GpuFunc::from_shader(
            device.clone(),
            &shader,
            "add_partial_sums",
        ));

        let values_buffer =
            InspectableBuffer::new_with_data(&device, "offsets_buffer", values.as_bytes())?;

        let (_, total_size_buffer) = add_partial_sums.call(&queue, &values_buffer)?;
        device.poll(PollType::wait_indefinitely())?;

        let values = values_buffer.read::<u32>(&device, &queue).await?;
        let total_size = total_size_buffer.read::<u32>(&device, &queue).await?[0];
        Ok((values, total_size))
    }

    #[tokio::test]
    async fn test_add_partial_sums() -> Result<()> {
        let mut values: Vec<u32> = (1..=256).cycle().take(1024).collect();
        for (prev, curr) in (1..=values.len() / 256)
            .map(|x| usize::try_from(x).unwrap() * 256 - 1)
            .tuple_windows::<(usize, usize)>()
        {
            values[curr] += values[prev];
        }
        eprintln!("values: {values:?}");
        let (result, total_size) = run_add_partial_sums_test(&values).await?;
        eprintln!("result: {result:?}");

        assert_eq!(result.len(), 1024, "result len");
        assert_eq!(total_size, 1024, "total size");
        for (idx, &elem) in result.iter().enumerate() {
            assert_eq!(idx + 1, usize::try_from(elem)?, "at index {idx}");
        }

        Ok(())
    }
}
